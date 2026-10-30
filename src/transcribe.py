"""Unchanged SraVaani defaults on final serialized WAV paths only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .audio import read_wav, sha256
from .data_manifest import read_manifest, save_json
from .runtime import cached_record, digest, environment, require_gpu, source_digest

ASR_REPO='ARTPARK-IISc/SraVaani-1.0'
ASR_REVISION='f5dd5358325a5208775b91dad98918e079ea2b27'
DECODER='official model.transcribe defaults, return_hypotheses=True'


def transcribe(manifest,audio_dir,output_dir,token=None):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    torch=require_gpu();torch.cuda.set_device(0)
    rows=read_manifest(manifest);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if not token: raise RuntimeError('Pass the existing Kaggle secret through stdin; never a command-line token')
    snapshot=Path(snapshot_download(ASR_REPO,revision=ASR_REVISION,token=token,allow_patterns=['*.ts','*.pt','*.model','*.json','*.py']))
    del token
    artifacts={p.name:sha256(p) for p in sorted(snapshot.iterdir()) if p.is_file() and p.suffix in ['.ts','.pt','.model','.json','.py']}
    config={'asr_repo':ASR_REPO,'asr_revision':ASR_REVISION,'decoder':DECODER,'artifact_sha256':artifacts,
            'environment':environment(),'source_hash':source_digest(),'device':'cuda:0','dtype':'official default (FP32 on this revision)'}
    config_hash=digest(config)
    prior=output_dir/'asr_config.json'
    if prior.exists() and digest(json.loads(prior.read_text()))!=config_hash:raise ValueError('ASR config changed; choose a new output directory')
    save_json(prior,config)
    # Load pinned local snapshot to avoid any unpinned remote-code dependency.
    start=time.perf_counter()
    model=AutoModel.from_pretrained(str(snapshot),trust_remote_code=True,local_files_only=True).to('cuda:0').eval()
    load_seconds=time.perf_counter()-start
    records={};texts={}
    for index, row in enumerate(rows, 1):
        path=Path(audio_dir)/row['filename'];read_wav(path,row['frames'])
        audio_hash=sha256(path);key=digest({'audio_sha256':audio_hash,'config_hash':config_hash})
        cache=output_dir/'cache'/(row['clip_id']+'.json')
        record=cached_record(cache,key,audio_hash)
        if record and record['text_sha256']!=hashlib.sha256(record['text'].encode()).hexdigest():
            raise ValueError('ASR cache text integrity mismatch')
        if not record:
            try:
                torch.cuda.reset_peak_memory_stats(0);torch.cuda.synchronize(0);start=time.perf_counter()
                # Do not alter model, decoder, text, spelling, case, or punctuation.
                hyps=model.transcribe(str(path),return_hypotheses=True)
                torch.cuda.synchronize(0);elapsed=time.perf_counter()-start
                if len(hyps)!=1 or not isinstance(hyps[0].text,str):raise ValueError('Unexpected ASR result')
                text=hyps[0].text
                if sha256(path)!=audio_hash:raise ValueError('WAV changed during transcription')
                record={'status':'success','clip_id':row['clip_id'],'cache_key':key,'audio_sha256':audio_hash,
                        'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'seconds':elapsed,
                        'audio_seconds':row['seconds'],'rtf':elapsed/row['seconds'],'peak_memory_bytes':torch.cuda.max_memory_allocated(0)}
                save_json(cache,record)
            except Exception as error:
                save_json(cache,{'status':'failed','clip_id':row['clip_id'],'cache_key':key,'error_type':type(error).__name__})
                raise
        records[row['clip_id']]=record;texts[row['clip_id']]=record['text']
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f'SraVaani: {index}/{len(rows)} clips complete', flush=True)
    path=output_dir/'transcripts.jsonl'
    path.write_text(''.join(json.dumps({'clip_id':cid,'text':texts[cid]},ensure_ascii=False)+'\n' for cid in sorted(texts)))
    provenance={**config,'transcripts_sha256':sha256(path),'clips':records}
    save_json(output_dir/'asr_provenance.json',provenance)
    elapsed=sum(r['seconds'] for r in records.values());audio_seconds=sum(r['audio_seconds'] for r in records.values())
    summary={'clips':len(records),'failures':0,'model_load_seconds':load_seconds,'inference_seconds':elapsed,'audio_seconds':audio_seconds,
             'rtf':elapsed/audio_seconds,'peak_memory_bytes':max(r['peak_memory_bytes'] for r in records.values())}
    summary['test_asr_wall_hours_estimate']=10.910046458333333*summary['rtf']
    save_json(output_dir/'asr_summary.json',summary);print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['manifest','audio-dir','output-dir']:p.add_argument('--'+n,required=True)
    import sys
    a=p.parse_args();transcribe(a.manifest,a.audio_dir,a.output_dir,token=sys.stdin.read().strip())
