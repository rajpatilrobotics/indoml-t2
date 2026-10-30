"""Paired training from GAPAdapt1; selection uses only external full-utterance metrics."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from src.audio import read_wav,write_wav,sha256
from src.data_manifest import save_json
from src.metrics import si_sdr,pooled_wer
from src.runtime import digest,source_digest,require_gpu
from experiments.gap_adapt_v1.data import Pairs,PINS
from experiments.gap_adapt_v1.train import hybrid_loss,checkpoint_blocks,save_state,restore_state
from experiments.predictive_v1.geometry import decode
from experiments.predictive_v1.vendor.tfgridnet import TFGridNet
from .teacher import Teacher,finite

RECIPE={'seed':20261010,'optimizer_steps':600,'accumulation':4,'crop_frames':64000,
        'pair_index_offset':4000,'learning_rate':1e-5,'weight_decay':.01,'betas':[.8,.99],
        'gradient_clip':3.,'eval_interval':200,'save_interval':25,'dtype':'FP32',
        'activation_checkpointing':'per GridNet block, non-reentrant',
        'arms':['control','perceptual'],'teacher_gradient_fraction':.3,'calibration_pairs':4,
        'teacher_weight_bounds':[.001,100.],
        'selection':'100 full external utterances; highest Combined with gain>=0.25, WER nonregression, SI loss<=0.5dB; 100 separate confirmation utterances then must improve Combined and not worsen WER.',
        'training_loss':'Same hybrid reconstruction in both arms. Perceptual arm additionally matches frozen SraVaani encoder cosine features; scalar calibrated once on training gradients.'}


def load(path): return json.loads(Path(path).read_text())


def experiment_hash():
    return digest({p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def initial_model(root,device):
    spec=load(Path(root)/'configs/phase2_gap_adapt.json')['model_input']['files']['training/selected_predictor.pt']
    p=Path(root)/'parent/training/selected_predictor.pt'
    if sha256(p)!=spec['sha256'] or p.stat().st_size!=spec['bytes']: raise ValueError('Continuation parent changed')
    state=torch.load(p,map_location='cpu',weights_only=True)
    expected=load(Path(root)/'parent/training/selection.json')['binding']['initialization']['config']
    if state['cfg']!=expected: raise ValueError('Parent architecture changed')
    model=TFGridNet(**state['cfg']);model.load_state_dict(state['model'],strict=True)
    checkpoint_blocks(model)
    return model.to(device),state['cfg']


def eligible(now,before,confirmation=False):
    gain=now['combined']-before['combined']
    return (gain>(0 if confirmation else .25-1e-12) and now['word_errors']<=before['word_errors']
            and now['si_sdr']>=before['si_sdr']-.5)


def choose_arm(summaries):
    ready=[s for s in summaries if s['selected']]
    if not ready:return None
    return sorted(ready,key=lambda s:(-s['best']['combined'],-s['best']['delta_wer_pp'],s['arm']))[0]['arm']


def tensors(pairs,index,device):
    x,y=pairs.get(index)
    return torch.from_numpy(x)[None].to(device),torch.from_numpy(y)[None].to(device)


def probe_pair(pairs,device):
    # A deliberately clean-preservation example cannot test a denoising-loss gradient.
    for index in range(RECIPE['pair_index_offset'],RECIPE['pair_index_offset']+32):
        x,y=tensors(pairs,index,device)
        if (x-y).abs().max().item()>1e-6:return x,y,index
    raise ValueError('No nonidentity training pair available for the gradient probe')


def calibrate(model,teacher,pairs,device):
    ratios=[];details=[];model.eval()
    for i in range(RECIPE['calibration_pairs']):
        x,y=tensors(pairs,RECIPE['pair_index_offset']+i,device)
        with torch.no_grad():pred=model(x,sr=16000)
        pred=pred.detach().requires_grad_(True)
        base=hybrid_loss(pred,y); perceptual=teacher.loss(pred,y)
        a,=torch.autograd.grad(base,pred,retain_graph=True);b,=torch.autograd.grad(perceptual,pred)
        finite(a,'reconstruction gradient');finite(b,'perceptual gradient')
        if min(a.norm().item(),b.norm().item())<=1e-12: raise ValueError('Invalid loss calibration gradient')
        ratio=RECIPE['teacher_gradient_fraction']*a.norm().item()/b.norm().item()
        ratios.append(ratio);details.append({'base_norm':a.norm().item(),'teacher_norm':b.norm().item()})
    weight=float(np.clip(statistics.median(ratios),*RECIPE['teacher_weight_bounds']))
    return {'weight':weight,'pairs':details,'policy':'Fixed median training-output gradient ratio, never validation tuned.'}


def update(model,optimizer,teacher,pairs,step,device,weight,accumulation=None):
    accumulation=accumulation or RECIPE['accumulation']
    model.train();optimizer.zero_grad(set_to_none=True);losses=[]
    for j in range(accumulation):
        x,y=tensors(pairs,RECIPE['pair_index_offset']+step*accumulation+j,device)
        pred=model(x,sr=16000);loss=hybrid_loss(pred,y)
        if weight: loss=loss+weight*teacher.loss(pred,y)
        finite(loss,'training loss');(loss/accumulation).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),RECIPE['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    if teacher and any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()): raise ValueError('Teacher was trained')
    return float(np.mean(losses)),float(norm)


def evaluate(root,model,teacher,rows,folder,checkpoint_hash):
    root,folder=Path(root),Path(folder);folder.mkdir(parents=True,exist_ok=True);model.eval()
    refs={r['clip_id']:r['transcript'] for r in rows}; noisy={}; enhanced={}; values=[];records={}
    binding={'checkpoint':checkpoint_hash,'manifest':sha256(root/'external_eval/manifest.json'),
             'experiment':experiment_hash(),'asr':load(root/'teacher_snapshot.json')['artifact_sha256']}
    for i,r in enumerate(rows,1):
        cid=r['clip_id'];x=root/'external_eval/noisy'/r['filename'];y=root/'external_eval/clean'/r['filename']
        if sha256(x)!=r['sha256'] or sha256(y)!=r['clean_sha256']: raise ValueError('External eval audio changed')
        cache=folder/(cid+'.json'); out=folder/'audio'/r['filename'];key=digest({'binding':binding,'row':r})
        record=load(cache) if cache.exists() else None
        if record and (record['key']!=key or not out.exists() or record['audio_sha256']!=sha256(out)): raise ValueError('Stale external evaluation cache')
        if not record:
            waveform=read_wav(x,r['frames'])
            def native(a):return model(torch.from_numpy(a.copy())[None].to(teacher.device),sr=16000)[0].cpu().numpy()
            with torch.inference_mode(): result=decode(waveform,native)
            receipt=write_wav(out,result,r['frames'])
            with torch.inference_mode():
                nt=teacher.model.transcribe(str(x),return_hypotheses=True)[0].text
                et=teacher.model.transcribe(str(out),return_hypotheses=True)[0].text
            if not isinstance(nt,str) or not isinstance(et,str): raise ValueError('Invalid external ASR result')
            record={'key':key,'audio_sha256':receipt['sha256'],'noisy_text':nt,'enhanced_text':et,
                    'si_sdr':si_sdr(read_wav(y),read_wav(out)),'binding':binding}
            save_json(cache,record)
        noisy[cid]=record['noisy_text'];enhanced[cid]=record['enhanced_text'];values.append(record['si_sdr']);records[cid]=record
        if i==1 or i%25==0 or i==len(rows): print(f'External development {folder.name}: {i}/{len(rows)}',flush=True)
    a=pooled_wer(refs,noisy,list(refs));b=pooled_wer(refs,enhanced,list(refs))
    if a['wer'] is None or b['wer'] is None or b['skipped_empty_references']:raise ValueError('External references incomplete')
    delta=100*(a['wer']-b['wer']); si=float(np.mean(values))
    report={'clips':len(rows),'combined':si+delta,'si_sdr':si,'delta_wer_pp':delta,'word_errors':b['edits'],
            'noisy_word_errors':a['edits'],'reference_words':b['reference_words'],'binding':binding,
            'source':'External Rasa full-utterance references, not competition validation or test'}
    save_json(folder/'metrics.json',report);print(json.dumps(report|{'binding':'saved separately'}),flush=True)
    return report


def main(root,arm):
    root=Path(root);require_gpu();torch.set_num_threads(2)
    if arm not in RECIPE['arms']:raise ValueError('Unknown arm')
    torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';folder=root/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    model,cfg=initial_model(root,device);teacher=Teacher(load(root/'teacher_snapshot.json')['path'],device)
    pairs=Pairs(root,'train');rows=[r for r in load(root/'external_eval/manifest.json') if r['partition']=='select']
    x,y,probe_index=probe_pair(pairs,device)
    probe=teacher.preflight(x,y);probe['training_pair_index']=probe_index;save_json(folder/'teacher_preflight.json',probe)
    calibration_path=folder/'calibration.json'
    calibration_binding={'experiment':experiment_hash(),'parent':sha256(root/'parent/training/selected_predictor.pt'),
                         'data':sha256(root/'adapt_sources.json'),'teacher':load(root/'teacher_snapshot.json')['artifact_sha256']}
    calibration=load(calibration_path) if calibration_path.exists() else calibrate(model,teacher,pairs,device)|{'binding':calibration_binding}
    if calibration['binding']!=calibration_binding:raise ValueError('Stale loss calibration')
    save_json(calibration_path,calibration);weight=calibration['weight'] if arm=='perceptual' else 0.
    binding={'recipe':RECIPE,'arm':arm,'teacher_weight':weight,'calibration':calibration,
             'data_sha256':sha256(root/'adapt_sources.json'),'data_pins_sha256':sha256(PINS),
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
        torch.cuda.reset_peak_memory_stats();loss,grad=update(model,optimizer,teacher,pairs,step,device,weight);step+=1
        if step==segment_step+1 or step%25==0:
            seconds=time.monotonic()-segment_start;speed=seconds/(step-segment_step)
            progress={'arm':arm,'optimizer_step':step,'total_steps':RECIPE['optimizer_steps'],'loss':loss,'gradient_norm':grad,
                      'peak_memory_bytes':torch.cuda.max_memory_allocated(),'seconds_per_update_including_dev':speed,
                      'remaining_training_hours_estimate_only':speed*(RECIPE['optimizer_steps']-step)/3600}
            save_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step%RECIPE['eval_interval']==0:
            ckpt=folder/f'step_{step:04d}.pt'
            save_state(ckpt,model,optimizer,step,binding,best,history)
            current=evaluate(root,model,teacher,rows,folder/f'dev_{step:04d}',sha256(ckpt))
            history.append({'step':step,'metrics':current})
            if eligible(current,history[0]['metrics']) and current['combined']>-best['loss']:
                best={'step':step,'loss':-current['combined'],'initial_loss':best['initial_loss']}
                torch.save({'cfg':cfg,'model':model.state_dict()},folder/f'predictor_{step:04d}.pt')
            save_json(folder/'history.json',history)
        if step%RECIPE['save_interval']==0:save_state(resume,model,optimizer,step,binding,best,history)
    if teacher.frozen_hash()!=probe['teacher_weights_sha256']:raise ValueError('Teacher changed during training/evaluation')
    selected=best['step']>0
    report={'arm':arm,'selected':selected,'best_step':best['step'],'completed_steps':step,
            'best':next(v['metrics'] for v in history if v['step']==best['step']),
            'baseline':history[0]['metrics'],'binding':binding,'history':history,'teacher_unchanged':True,
            'elapsed_seconds':time.monotonic()-started,
            'selected_checkpoint_sha256':sha256(folder/f"predictor_{best['step']:04d}.pt") if selected else None}
    save_json(folder/'summary.json',report);print('ARM COMPLETE',arm,'selected:',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=RECIPE['arms'],required=True);a=p.parse_args();main(Path.cwd(),a.arm)
