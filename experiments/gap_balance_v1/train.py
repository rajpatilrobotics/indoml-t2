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
from experiments.gap_adapt_v1.train import checkpoint_blocks
from experiments.predictive_v1.vendor.tfgridnet import TFGridNet
from experiments.gap_token_pilot_v5.teacher import Teacher, OBJECTIVE, finite
from experiments.gap_token_pilot_v5.diagnostics import failure_evidence
from experiments.predictive_v1.geometry import decode
from .objective import reconstruction,combine_gradients
from experiments.gap_token_pilot_v5.data import Pairs,PINS,ARMS

RECIPE={'seed':20261011,'optimizer_steps':600,'accumulation':4,'crop_frames':64000,
 'pair_index_offset':2400,'learning_rate':1e-5,'weight_decay':.01,'betas':[.8,.99],
 'gradient_clip':3.,'eval_interval':200,'save_interval':25,'dtype':'FP32',
 'activation_checkpointing':'Enhancer per GridNet block and frozen ASR encoder, non-reentrant',
 'arms':list(ARMS),'identity_fraction':.2,
 'curriculum':'Both arms: 40% SNR -15..0; 40% SNR 0..20; 20% clean preservation',
 'teacher_gradient_fraction':.5,'balance_space':'model parameters, each microbatch','max_recognition_gradient_ratio':.5,
 'teacher_objective':OBJECTIVE,'teacher_encoder_backend':'Native PyTorch, JIT optimization off within training encode only; scoring transcribe unchanged','startup_diagnostics':'Three original checkpointed backward checks, anomaly traces, per-worker startup; no mathematical objective change',
 'selection':'Best eligible200/400/600 per arm on reused external select240; select one arm, then reused confirm240. Require noisy SI loss<=0.1dB and clean/WER guards. Full2501 validation gain>=0.5 required for test trial.',
 'training_loss':'Both: hybrid reconstruction +0.1 smooth capped negative SI-SDR(30dB). Token: frozen clean-path KL, one-sided parameter gradient conflict projection and recognition norm cap0.5 relative to reconstruction. Raw-gradient constraint is not an AdamW or generalization guarantee.',
 'data_scope':'Identical paired crops2400..4799, disjoint training/dev source IDs; external evaluation reused and not a fresh holdout.'}



def initial_model(root,device):
    root=Path(root); spec=load(root/'configs/phase2_gap_noise.json')['model_input']['files']['training/selected_predictor.pt']
    path=root/'parent/training/selected_predictor.pt'
    if sha256(path)!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise ValueError('GAPNoise1 parent changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    cfg=load(root/'parent/training/selection.json')['binding']['initialization']['config']
    if state['cfg']!=cfg:raise ValueError('Parent architecture changed')
    model=TFGridNet(**cfg);model.load_state_dict(state['model'],strict=True);checkpoint_blocks(model)
    return model.to(device),cfg


def calibration_binding(root):
    root=Path(root)
    return {'experiment':experiment_hash(),'parent':sha256(root/'parent/training/selected_predictor.pt'),
            'data':sha256(root/'adapt_sources.json'),'teacher':load(root/'teacher_snapshot.json')['artifact_sha256']}


def preflight(root):
    root=Path(root);require_gpu();torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,cfg=initial_model(root,'cuda:0');teacher=Teacher(load(root/'teacher_snapshot.json')['path'],'cuda:0')
    pairs=Pairs(root,'token');x,y=pairs.get(RECIPE['pair_index_offset']+2)
    x=torch.from_numpy(x)[None].cuda();y=torch.from_numpy(y)[None].cuda()
    report=teacher.preflight(x,y,root/'numeric_diagnostics/teacher')
    frozen=teacher.frozen_hash();model.train();params=tuple(model.parameters())
    details=[]
    # Actual full-network gradients, no optimizer updates, covering clean/hard/moderate.
    for index in (0,2,6):
        x,y=pairs.get(RECIPE['pair_index_offset']+index)
        x=torch.from_numpy(x)[None].cuda();y=torch.from_numpy(y)[None].cuda()
        with failure_evidence(root/'numeric_diagnostics'/('balance_'+str(index)),{'noisy':x,'clean':y},{'stage':'parameter_gradient_preflight'},anomaly=True):
            pred=model(x,sr=16000);rec=reconstruction(pred,y);kd=teacher.loss(pred,y)
            rg=torch.autograd.grad(rec,params,retain_graph=True)
            kg=torch.autograd.grad(kd,params)
            combined,info=combine_gradients(rg,kg,RECIPE['max_recognition_gradient_ratio'])
            if not all(torch.isfinite(g).all() for g in combined):raise ValueError('Nonfinite combined gradient')
            details.append(dict(index=index,reconstruction_loss=float(rec.detach()),posterior_KL=float(kd.detach()),**info))
        del pred,rec,kd,rg,kg,combined
    if frozen!=teacher.frozen_hash() or any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()):raise ValueError('Teacher changed')
    calibration={'weight':1.,'policy':'No fixed scalar calibration; project and cap actual parameter gradients each microbatch.', 'binding':calibration_binding(root)}
    report.update(status='PASS',binding=calibration['binding'],optimizer_updates=0,parameter_gradient_checks=details)
    save_json(root/'token_calibration.json',calibration);save_json(root/'token_preflight.json',report)
    print('BALANCED GRADIENT PREFLIGHT PASSED',json.dumps(report),flush=True)


def load(path):return json.loads(Path(path).read_text())


def experiment_hash():
    return digest({p.name:sha256(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json')})


def eligible(now,before,confirmation=False):
    gain=now['combined']-before['combined']
    return (gain>(0 if confirmation else .25-1e-12)
            and now['word_errors']<=before['word_errors'] and now['si_sdr']>=before['si_sdr']-.1
            and now['clean']['word_errors']<=before['clean']['word_errors']
            and now['clean']['si_sdr']>=before['clean']['si_sdr']-.5)


def choose_arm(summaries):
    ready=[s for s in summaries if s['selected']]
    return sorted(ready,key=lambda s:(-s['best']['combined'],-s['best']['delta_wer_pp'],s['arm']))[0]['arm'] if ready else None


def update(model,optimizer,pairs,step,device,teacher=None,weight=0.,accumulation=None,diagnostic_folder=None):
    accumulation=accumulation or RECIPE['accumulation']
    model.train();optimizer.zero_grad(set_to_none=True);params=tuple(model.parameters())
    values=[];posterior=[];details=[]
    try:
        for j in range(accumulation):
            x,y=pairs.get(RECIPE['pair_index_offset']+step*accumulation+j)
            x=torch.from_numpy(x)[None].to(device);y=torch.from_numpy(y)[None].to(device)
            pred=model(x,sr=16000);rec=reconstruction(pred,y);finite(rec,'reconstruction loss')
            if weight:
                if teacher is None:raise ValueError('Missing frozen teacher')
                kd=teacher.loss(pred,y);finite(kd,'posterior loss')
                rg=torch.autograd.grad(rec,params,retain_graph=True)
                kg=torch.autograd.grad(kd,params)
                grads,info=combine_gradients(rg,kg,RECIPE['max_recognition_gradient_ratio'])
                posterior.append(float(kd.detach()));details.append(info)
                del rg,kg,kd
            else:
                grads=torch.autograd.grad(rec,params)
            for p,g in zip(params,grads):
                finite(g,'parameter gradient')
                if p.grad is None:p.grad=g.detach().clone()/accumulation
                else:p.grad.add_(g.detach(),alpha=1./accumulation)
            values.append(float(rec.detach()));del pred,rec,grads
        norm=torch.nn.utils.clip_grad_norm_(params,RECIPE['gradient_clip'],error_if_nonfinite=True)
        if float(norm)<=0:raise ValueError('No enhancement gradient')
        if teacher is not None and any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()):raise ValueError('Teacher changed')
        optimizer_called=True;optimizer.step()
        return float(np.mean(values)),float(norm),{
            'reconstruction_loss':float(np.mean(values)),
            'posterior_KL':float(np.mean(posterior)) if posterior else None,
            'balanced_parameter_gradients':details,'gradient_clip_scale':min(1.,RECIPE['gradient_clip']/(float(norm)+1e-6))}
    except Exception as error:
        if diagnostic_folder is not None:
            from experiments.gap_token_pilot_v5.diagnostics import preserve_error
            tensors={k:v for k,v in locals().items() if k in ('x','y','pred') and isinstance(v,torch.Tensor)}
            preserve_error(diagnostic_folder,error,tensors,{'stage':'balanced_update','attempted_step':step+1,'optimizer_step_may_have_started':locals().get('optimizer_called',False)})
        raise


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
    x0,y0=pairs.get(2);x0=torch.from_numpy(x0)[None].to(next(model.parameters()).device);y0=torch.from_numpy(y0)[None].to(next(model.parameters()).device)
    with failure_evidence(folder/'numeric_diagnostics/startup', {'noisy':x0,'clean':y0}, {'stage':'fresh_worker','arm':arm}, anomaly=True):
        startup=teacher.preflight(x0,y0,diagnostic_folder=folder/'numeric_diagnostics/teacher')
    save_json(folder/'startup_preflight.json',startup)
    del x0,y0
    calibration=load(root/'token_calibration.json');probe=load(root/'token_preflight.json')
    if calibration['binding']!=calibration_binding(root) or probe['binding']!=calibration['binding'] or probe['status']!='PASS':raise ValueError('Stale/missing token preflight')
    weight=calibration['weight'] if arm=='token' else 0.
    rows=[r for r in load(root/'external_eval/manifest.json') if r['partition']=='select']
    binding={'recipe':RECIPE,'arm':arm,'teacher_weight':weight,'calibration':calibration,'data_sha256':sha256(root/'adapt_sources.json'),'data_pins_sha256':sha256(PINS),
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
    segment_step=step;segment_start=time.monotonic();update_seconds=0.;update_times=[]
    timing_path=folder/'timing.json'
    timings=load(timing_path)['updates'] if timing_path.exists() else []
    # A crash may leave timing ahead of the atomic optimizer checkpoint.
    timings=[item for item in timings if item['step']<=step]
    while step<RECIPE['optimizer_steps']:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();update_start=time.monotonic()
        loss,grad,components=update(model,optimizer,pairs,step,device,teacher,weight,diagnostic_folder=folder/'numeric_diagnostics/update');torch.cuda.synchronize()
        seconds=time.monotonic()-update_start;update_times.append(seconds);update_seconds+=seconds;step+=1
        timings.append({'step':step,'seconds':seconds,'components':components,
                        'peak_memory_bytes':torch.cuda.max_memory_allocated()})
        if step==segment_step+1 or step%25==0:
            speed=(time.monotonic()-segment_start)/(step-segment_step)
            progress={'arm':arm,'optimizer_step':step,'total_steps':RECIPE['optimizer_steps'],'loss':loss,'gradient_norm':grad,
              'peak_memory_bytes':torch.cuda.max_memory_allocated(),'seconds_per_update_including_dev':speed,
              'remaining_training_hours_estimate_only':speed*(RECIPE['optimizer_steps']-step)/3600,
              'update_only_seconds_mean':update_seconds/(step-segment_step),'components':components}
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
        if step%RECIPE['save_interval']==0:
            save_json(timing_path,{'updates':timings})
            save_state(resume,model,optimizer,step,binding,best,history)
    if teacher.frozen_hash()!=teacher_hash or any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()):raise ValueError('ASR changed')
    selected=best['step']>0
    report={'arm':arm,'selected':selected,'best_is_diagnostic_only':False,'best_step':best['step'],'completed_steps':step,
      'best':next(v['metrics'] for v in history if v['step']==best['step']),'baseline':history[0]['metrics'],
      'binding':binding,'history':history,'teacher_unchanged':True,'elapsed_seconds':time.monotonic()-started,
      'timing':{'updates_this_session':len(update_times),'recorded_updates_total':len(timings),
                'update_seconds':[item['seconds'] for item in timings],
                'update_mean_seconds':float(np.mean([item['seconds'] for item in timings])) if timings else None,
                'scope':'Actual synchronized training only; excludes setup/evaluation.'},
      'selected_checkpoint_sha256':sha256(folder/f"predictor_{best['step']:04d}.pt") if selected else None}
    save_json(folder/'summary.json',report);print('ARM COMPLETE',arm,'selected:',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:
        try:preflight(Path.cwd())
        except Exception as error:
            import traceback
            save_json('token_preflight_error.json',{'status':'failed','error_type':type(error).__name__,
                      'message':str(error),'traceback':traceback.format_exc(),'training_started':False})
            raise
    elif a.arm:main(Path.cwd(),a.arm)
    else:p.error('Choose --preflight or --arm')
