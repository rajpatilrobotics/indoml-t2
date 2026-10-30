"""Paired data curricula, with noisy utility and a separate clean speech guard."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.audio import read_wav,write_wav,sha256
from src.data_manifest import save_json
from src.metrics import si_sdr,pooled_wer
from src.runtime import digest,source_digest,require_gpu
from experiments.gap_adapt_v1.train import hybrid_loss,save_state,restore_state
from experiments.gap_asr_v1.train import initial_model
from experiments.gap_asr_v1.teacher import Teacher
from experiments.predictive_v1.geometry import decode
from .data import Pairs,PINS,ARMS

RECIPE={'seed':20261010,'optimizer_steps':800,'accumulation':4,'crop_frames':64000,
 'pair_index_offset':0,'learning_rate':1e-5,'weight_decay':.01,'betas':[.8,.99],
 'gradient_clip':3.,'eval_interval':400,'save_interval':25,'dtype':'FP32',
 'activation_checkpointing':'per GridNet block, non-reentrant','arms':list(ARMS),
 'identity_fraction':.2,'standard_snr_db':[-5,20],
 'hard_curriculum':'40% SNR -15..0; 40% SNR 0..20; 20% clean preservation',
 'noise':'Two distinct external recordings per pair; second relative level -10..0 dB. Identical draws across arms.',
 'selection':'200 noisy selection utterances: Combined gain >=0.25, WER nonregression, SI loss<=0.5dB. 40 clean utterances: WER nonregression and SI loss<=0.5dB. Repeat once on disjoint confirmation, requiring strictly positive noisy Combined gain.',
 'training_loss':'Same reviewed hybrid reconstruction in both arms; no recognition-feature loss or ASR updates.'}


def load(path):return json.loads(Path(path).read_text())


def experiment_hash():
    return digest({p.name:sha256(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json')})


def eligible(now,before,confirmation=False):
    gain=now['combined']-before['combined']
    return (gain>(0 if confirmation else .25-1e-12)
            and now['word_errors']<=before['word_errors'] and now['si_sdr']>=before['si_sdr']-.5
            and now['clean']['word_errors']<=before['clean']['word_errors']
            and now['clean']['si_sdr']>=before['clean']['si_sdr']-.5)


def choose_arm(summaries):
    ready=[s for s in summaries if s['selected']]
    return sorted(ready,key=lambda s:(-s['best']['combined'],-s['best']['delta_wer_pp'],s['arm']))[0]['arm'] if ready else None


def update(model,optimizer,pairs,step,device,accumulation=None):
    accumulation=accumulation or RECIPE['accumulation']
    model.train();optimizer.zero_grad(set_to_none=True);values=[]
    for j in range(accumulation):
        x,y=pairs.get(step*accumulation+j)
        x=torch.from_numpy(x)[None].to(device);y=torch.from_numpy(y)[None].to(device)
        loss=hybrid_loss(model(x,sr=16000),y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        (loss/accumulation).backward();values.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE['gradient_clip'],error_if_nonfinite=True)
    if float(norm)<=0:raise ValueError('No enhancement gradient')
    optimizer.step()
    return float(np.mean(values)),float(norm)


def summarize(rows,records):
    def group(selected):
        refs={r['clip_id']:r['transcript'] for r in selected};ids=list(refs)
        a=pooled_wer(refs,{i:records[i]['noisy_text'] for i in ids},ids)
        b=pooled_wer(refs,{i:records[i]['enhanced_text'] for i in ids},ids)
        if a['wer'] is None or b['wer'] is None or b['skipped_empty_references']:raise ValueError('Incomplete references')
        delta=100*(a['wer']-b['wer']);si=float(np.mean([records[i]['si_sdr'] for i in ids]))
        if not np.isfinite(si):raise ValueError('Nonfinite external metric')
        return {'clips':len(ids),'combined':si+delta,'si_sdr':si,'delta_wer_pp':delta,
                'word_errors':b['edits'],'noisy_word_errors':a['edits'],'reference_words':b['reference_words']}
    report=group([r for r in rows if r['condition']!='clean'])
    report['clean']=group([r for r in rows if r['condition']=='clean'])
    report['strata']={k:group([r for r in rows if r['condition']==k]) for k in ('hard','moderate')}
    report['total_clips']=len(rows)
    return report


def evaluate(root,model,teacher,rows,folder,checkpoint_hash):
    root,folder=Path(root),Path(folder);folder.mkdir(parents=True,exist_ok=True);model.eval();records={}
    binding={'checkpoint':checkpoint_hash,'manifest':sha256(root/'external_eval/manifest.json'),
             'experiment':experiment_hash(),'asr':load(root/'teacher_snapshot.json')['artifact_sha256']}
    for i,r in enumerate(rows,1):
        cid=r['clip_id'];x=root/'external_eval/noisy'/r['filename'];y=root/'external_eval/clean'/r['filename']
        if sha256(x)!=r['sha256'] or sha256(y)!=r['clean_sha256']:raise ValueError('External audio changed')
        cache=folder/(cid+'.json');out=folder/'audio'/r['filename'];key=digest({'binding':binding,'row':r})
        record=load(cache) if cache.exists() else None
        if record and (record['key']!=key or not out.exists() or record['audio_sha256']!=sha256(out)):raise ValueError('Stale evaluation cache')
        if not record:
            waveform=read_wav(x,r['frames'])
            def native(a):return model(torch.from_numpy(a.copy())[None].to(teacher.device),sr=16000)[0].cpu().numpy()
            with torch.inference_mode():result=decode(waveform,native)
            receipt=write_wav(out,result,r['frames'])
            with torch.inference_mode():
                nt=teacher.model.transcribe(str(x),return_hypotheses=True)[0].text
                et=teacher.model.transcribe(str(out),return_hypotheses=True)[0].text
            if not isinstance(nt,str) or not isinstance(et,str):raise ValueError('Invalid ASR output')
            record={'key':key,'audio_sha256':receipt['sha256'],'noisy_text':nt,'enhanced_text':et,
                    'si_sdr':si_sdr(read_wav(y),read_wav(out)),'binding':binding}
            save_json(cache,record)
        records[cid]=record
        if i==1 or i%25==0 or i==len(rows):print(f'External {folder.name}: {i}/{len(rows)}',flush=True)
    report=summarize(rows,records)|{'binding':binding,'source':'External Rasa evaluation; noisy-only Combined with separate clean preservation'}
    save_json(folder/'metrics.json',report);print(json.dumps(report|{'binding':'saved separately'}),flush=True)
    return report


def main(root,arm):
    root=Path(root);require_gpu();torch.set_num_threads(2)
    if arm not in ARMS:raise ValueError('Unknown arm')
    torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';folder=root/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    model,cfg=initial_model(root,device);teacher=Teacher(load(root/'teacher_snapshot.json')['path'],device)
    teacher_hash=teacher.frozen_hash();pairs=Pairs(root,arm)
    rows=[r for r in load(root/'external_eval/manifest.json') if r['partition']=='select']
    binding={'recipe':RECIPE,'arm':arm,'data_sha256':sha256(root/'adapt_sources.json'),'data_pins_sha256':sha256(PINS),
       'parent_checkpoint':sha256(root/'parent/training/selected_predictor.pt'),
       'initialization':load(root/'parent/training/selection.json')['binding']['initialization'],
       'source_hash':source_digest(),'experiment_hash':experiment_hash(),
       'external_manifest_sha256':sha256(root/'external_eval/manifest.json'),
       'teacher_artifact_sha256':load(root/'teacher_snapshot.json')['artifact_sha256']}
    optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['learning_rate'],betas=tuple(RECIPE['betas']),weight_decay=RECIPE['weight_decay'])
    started=time.monotonic();resume=folder/'resume.pt'
    if resume.exists():
        state=restore_state(resume,model,optimizer,binding);step=state['step'];best=state['best'];history=state['history']
    else:
        baseline=evaluate(root,model,teacher,rows,folder/'dev_0000',binding['parent_checkpoint'])
        step=0;best={'step':0,'loss':-baseline['combined'],'initial_loss':-baseline['combined']};history=[{'step':0,'metrics':baseline}]
        torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
        save_state(resume,model,optimizer,step,binding,best,history)
    segment_step=step;segment_start=time.monotonic()
    while step<RECIPE['optimizer_steps']:
        torch.cuda.reset_peak_memory_stats();loss,grad=update(model,optimizer,pairs,step,device);step+=1
        if step==segment_step+1 or step%25==0:
            speed=(time.monotonic()-segment_start)/(step-segment_step)
            progress={'arm':arm,'optimizer_step':step,'total_steps':RECIPE['optimizer_steps'],'loss':loss,'gradient_norm':grad,
              'peak_memory_bytes':torch.cuda.max_memory_allocated(),'seconds_per_update_including_dev':speed,
              'remaining_training_hours_estimate_only':speed*(RECIPE['optimizer_steps']-step)/3600}
            save_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step%RECIPE['eval_interval']==0:
            ckpt=folder/f'step_{step:04d}.pt';save_state(ckpt,model,optimizer,step,binding,best,history)
            predictor=folder/f'predictor_{step:04d}.pt'
            # Bind metrics directly to exported model-only weights, including rejected checkpoints.
            torch.save({'cfg':cfg,'model':model.state_dict()},predictor)
            now=evaluate(root,model,teacher,rows,folder/f'dev_{step:04d}',sha256(predictor));history.append({'step':step,'metrics':now})
            if eligible(now,history[0]['metrics']) and now['combined']>-best['loss']:
                best={'step':step,'loss':-now['combined'],'initial_loss':best['initial_loss']}
            save_json(folder/'history.json',history)
        if step%RECIPE['save_interval']==0:save_state(resume,model,optimizer,step,binding,best,history)
    if teacher.frozen_hash()!=teacher_hash or any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()):raise ValueError('ASR changed')
    selected=best['step']>0
    report={'arm':arm,'selected':selected,'best_step':best['step'],'completed_steps':step,
      'best':next(v['metrics'] for v in history if v['step']==best['step']),'baseline':history[0]['metrics'],
      'binding':binding,'history':history,'teacher_unchanged':True,'elapsed_seconds':time.monotonic()-started,
      'selected_checkpoint_sha256':sha256(folder/f"predictor_{best['step']:04d}.pt") if selected else None}
    save_json(folder/'summary.json',report);print('ARM COMPLETE',arm,'selected:',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();main(Path.cwd(),a.arm)
