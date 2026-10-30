#!/usr/bin/env python3
"""Replay the selected training lineage, with original update functions and resumable state.

This driver replays fixed selected recipes; it does not repeat historical candidate
search or promise byte-identical GPU optimization. Use the released checkpoint
for reproduction of the submitted inference pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from src.audio import sha256
from src.data_manifest import save_json
from src.runtime import digest, require_gpu
from experiments.gap_adapt_v1.train import checkpoint_blocks, load_initial, save_state, restore_state
from experiments.predictive_v1.vendor.tfgridnet import TFGridNet

ROOT=Path(__file__).resolve().parent


def prepare(root, stage, parent, token):
    if stage=='balance':
        for name in ('adapt_sources.json','adapt_data','external_eval'):
            src=parent/name;dest=root/name
            if not src.exists():raise FileNotFoundError(src)
            if not dest.exists():dest.symlink_to(src.resolve(),target_is_directory=src.is_dir())
        return
    if stage=='adapt':
        from experiments.gap_adapt_v1.data import prepare as fn
    else:
        from experiments.gap_noise_v1.data import prepare as fn
        (root/'parent').mkdir(exist_ok=True)
        dest=root/'parent/adapt_sources.json'
        if dest.exists() and sha256(dest)!=sha256(parent/'adapt_sources.json'):
            raise ValueError('Different parent data')
        shutil.copyfile(parent/'adapt_sources.json',dest)
    fn(root,token)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['adapt','noise','balance'],required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--parent-run',type=Path)
    p.add_argument('--prepare-data',action='store_true')
    a=p.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=True)
    parent=a.parent_run.resolve() if a.parent_run else None
    if a.stage!='adapt' and parent is None:p.error('--parent-run is required')
    if a.prepare_data:
        prepare(root,a.stage,parent,os.environ.get('HF_TOKEN'));return
    require_gpu();torch.set_num_threads(2)
    if a.stage=='adapt':
        from experiments.gap_adapt_v1 import train as impl
        from experiments.gap_adapt_v1.data import Pairs
        pairs=Pairs(root,'train')
    elif a.stage=='noise':
        from experiments.gap_noise_v1 import train as impl
        from experiments.gap_noise_v1.data import Pairs
        pairs=Pairs(root,'hard')
    else:
        from experiments.gap_balance_v1 import train as impl
        from experiments.gap_token_pilot_v5.data import Pairs
        pairs=Pairs(root,'token')
    recipe=impl.RECIPE
    torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='adapt':model,cfg,origin=load_initial('cuda:0')
    else:
        path=parent/'selected_predictor.pt'
        state=torch.load(path,map_location='cpu',weights_only=True);cfg=state['cfg']
        model=TFGridNet(**cfg);model.load_state_dict(state['model'],strict=True);model.cuda()
        origin={'sha256':sha256(path)}
    checkpoint_blocks(model)
    teacher=None
    if a.stage=='balance':
        from experiments.gap_token_check_v1.snapshot import prepare_snapshot
        from experiments.gap_token_pilot_v5.teacher import Teacher
        spec=json.loads((ROOT/'configs/asr.json').read_text())
        save_json(root/'configs/phase2_gap_adapt.json',{'validated_asr_config':spec})
        folder=prepare_snapshot(root,os.environ.get('HF_TOKEN'))
        teacher=Teacher(folder,'cuda:0')
        x,y=pairs.get(recipe['pair_index_offset']+2)
        teacher.preflight(torch.from_numpy(x)[None].cuda(),torch.from_numpy(y)[None].cuda(),root/'numeric_diagnostics')
    # Reset after preparation, as in the original workers before training updates.
    torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],betas=tuple(recipe['betas']),weight_decay=recipe['weight_decay'])
    source={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'experiments').rglob('*.py')}
    binding={'stage':a.stage,'recipe':recipe,'initialization':origin,'data_sha256':sha256(root/'adapt_sources.json'),'source':digest(source),'driver':sha256(Path(__file__))}
    latest=root/'resume.pt';history=[];step=0
    if latest.exists():
        state=restore_state(latest,model,optimizer,binding);step=state['step'];history=state['history']
    while step<recipe['optimizer_steps']:
        if a.stage=='balance':
            loss,norm,components=impl.update(model,optimizer,pairs,step,'cuda:0',teacher=teacher,weight=1.,diagnostic_folder=root/'numeric_diagnostics')
        else:
            loss,norm=impl.update(model,optimizer,pairs,step,'cuda:0',accumulation=recipe['accumulation']);components={}
        step+=1
        if step==1 or step%25==0:
            row={'step':step,'loss':loss,'gradient_norm':norm};history.append(row);print(json.dumps(row),flush=True)
            save_state(latest,model,optimizer,step,binding,{},history)
        if step%200==0 or step==recipe['optimizer_steps']:
            tmp=root/'predictor.partial.pt';torch.save({'cfg':cfg,'model':model.state_dict()},tmp)
            tmp.replace(root/f'predictor_{step:04d}.pt')
    checkpoint=root/'selected_predictor.pt'
    selected=root/f'predictor_{step:04d}.pt'
    if checkpoint.exists() and sha256(checkpoint)!=sha256(selected):raise ValueError('Different existing selected checkpoint')
    if not checkpoint.exists():shutil.copyfile(selected,checkpoint)
    save_json(root/'TRAIN_COMPLETED.json',{'status':'complete','stage':a.stage,'updates':step,'checkpoint_sha256':sha256(checkpoint),'selection':'Fixed replay of historically selected stage; no new validation score asserted','binding':binding})

if __name__=='__main__':main()
