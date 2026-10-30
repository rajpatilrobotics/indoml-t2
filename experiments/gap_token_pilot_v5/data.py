"""Broader pinned sources, fresh external holdouts and paired SNR curricula."""
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from src.audio import read_wav, write_wav, sha256
from src.data_manifest import save_json, read_manifest
from src.runtime import digest
from experiments.gap_adapt_v1.data import (crop, mixture, check_splits, pcm_hash,
    pcm_wav_hash, valid_speech, decode_clean, validate_noise_pins)

PINS = Path(__file__).with_name('data_pins.json')
ARMS = ('control', 'token')
RATE = 16000


def verify_shard(path, shard):
    if path.stat().st_size != shard['bytes']: raise ValueError('Rasa shard size mismatch')
    value = sha256(path)
    if 'sha256' in shard:
        if value != shard['sha256']: raise ValueError('Rasa shard hash mismatch')
    else:
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{value}\nsize {shard['bytes']}\n".encode()
        obj = hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest()
        if obj != shard['git_lfs_pointer_sha1']: raise ValueError('Rasa pinned LFS object mismatch')
    return value


def prepare_noise(root, noise, excluded_hashes):
    root = Path(root); validate_noise_pins(noise); rows = []
    save_json(root / 'adapt_noise_verified.json', {'status': 'checking', 'sources': 0})
    for i, f in enumerate(noise['files'], 1):
        print('Checking noise:', f['path'], f'({i}/{len(noise["files"])})', flush=True)
        save_json(root / 'adapt_data_stage.json', {'stage': 'noise', 'source': f['path'], 'index': i, 'total': len(noise['files'])})
        out = root / 'adapt_data/noise' / f['path'].replace('/', '__')
        if out.exists():
            raw = out.read_bytes()
        else:
            url = 'https://raw.githubusercontent.com/' + noise['repo'] + '/' + noise['revision'] + '/' + f['path']
            with urllib.request.urlopen(url, timeout=90) as response:
                raw = response.read()
        blob_hash = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        h = hashlib.sha256(raw).hexdigest()
        if len(raw) != f['bytes'] or blob_hash != f['git_blob_sha1'] or h != f['sha256']:
            raise ValueError('Pinned noise source mismatch: ' + f['path'])
        x = read_wav(raw); decoded_hash = pcm_hash(x)
        if decoded_hash != f['pcm_sha256'] or len(x) != f['frames']:
            raise ValueError('Pinned noise PCM mismatch: ' + f['path'])
        if len(x) < 4 * RATE or np.std(x) < 1e-5:
            raise ValueError('Unusable noise source: ' + f['path'])
        if h in excluded_hashes:
            raise ValueError('Noise overlaps competition data: ' + f['path'])
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists(): out.write_bytes(raw)
        rows.append({'kind': 'noise', 'split': f['split'], 'group': f['category'], 'source_id': f['path'],
                     'source_sha256': h, 'audio_sha256': h, 'pcm_sha256': decoded_hash,
                     'path': str(out.relative_to(root)), 'frames': len(x)})
    save_json(root / 'adapt_noise_verified.json', {'status': 'PASS', 'sources': len(rows),
                                                 'encoded_and_decoded_unique': True, 'rows': rows})
    return rows


def condition(index, arm, rng):
    if arm not in ARMS: raise ValueError('Unknown curriculum')
    identity = index % 10 < 2
    bounds = (-15., 0.) if index % 10 < 6 else (0., 20.)
    return identity, float(rng.uniform(*bounds))


def combined_noise(root, sources, frames, rng):
    """Same two-recording noise mixture for both arms, never competition noise."""
    ids = rng.choice(len(sources), size=2, replace=False)
    waves = []
    for i in ids:
        x = read_wav(Path(root)/sources[int(i)]['path'])
        if len(x) < frames: x = np.tile(x, 1+frames//len(x))
        x = crop(x, frames, rng); x -= x.mean()
        rms = float(np.sqrt(np.mean(x*x)))
        if rms < 1e-6: raise ValueError('Silent noise crop')
        waves.append(x/rms)
    return waves[0] + 10**(float(rng.uniform(-10, 0))/20)*waves[1]


def evaluation_assignment(index):
    # In each language: 200 noisy + 40 clean, divided 100+20 per partition.
    position = index % 12
    return ('select' if (index//12)%2 == 0 else 'confirm',
            'clean' if position < 2 else 'hard' if position < 7 else 'moderate')


def check_exclusions(rows, parent_rows, holdout):
    check_splits(rows)
    external_ids = [r['source_id'] for r in rows if r['kind']=='speech']
    if len(external_ids) != len(set(external_ids)): raise ValueError('Duplicate external speech ID')
    forbidden_ids = {r['clip_id'] for r in holdout}
    forbidden_hashes = {r['sha256'] for r in holdout} | {r['clean_sha256'] for r in holdout if 'clean_sha256' in r}
    for r in rows:
        if r['source_id'] in forbidden_ids or r['audio_sha256'] in forbidden_hashes or r['source_sha256'] in forbidden_hashes:
            raise ValueError('Competition/external overlap')
    prior_ids = {r['source_id'] for r in parent_rows if r['kind']=='speech'}
    prior_pcm = {r['pcm_sha256'] for r in parent_rows if r['kind']=='speech'}
    if any(r['split']=='dev' and (r['source_id'] in prior_ids or r['pcm_sha256'] in prior_pcm) for r in rows if r['kind']=='speech'):
        raise ValueError('External evaluation must exclude previous pilot speech')


def prepare(root, token):
    root=Path(root); pins=json.loads(PINS.read_text()); validate_noise_pins(pins['noise'])
    sys.path.insert(0, '/kaggle/temp/adapt_arrow')
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    holdout=read_manifest(root/'data/manifests/validation.jsonl')+read_manifest(root/'data/manifests/test.jsonl')
    excluded_ids={r['clip_id'] for r in holdout}
    excluded_hashes={r['sha256'] for r in holdout}|{r['clean_sha256'] for r in holdout if 'clean_sha256' in r}
    parent=json.loads((root/'parent/adapt_sources.json').read_text())['rows']
    parent += json.loads((Path(__file__).with_name('prior_speech_exclusions.json')).read_text())['rows']
    prior_ids={r['source_id'] for r in parent if r['kind']=='speech'}
    prior_pcm={r['pcm_sha256'] for r in parent if r['kind']=='speech'}
    rows=prepare_noise(root,pins['noise'],excluded_hashes)
    seen={r['audio_sha256'] for r in rows}; seen_pcm={r['pcm_sha256'] for r in rows}; speech_ids=set(); rejected=Counter()
    for shard in pins['rasa']['shards']:
        used_before=sum(r['kind']=='speech' and r['split']=='dev' and r.get('language')==shard['language'] for r in rows)
        limit=shard['limit'] if shard['split']=='train' else max(0, 240-used_before)
        if not limit: continue
        print('Checking pinned Rasa shard:',shard['path'],flush=True)
        path=Path(hf_hub_download(pins['rasa']['repo'],shard['path'],repo_type='dataset',revision=pins['rasa']['revision'],token=token))
        verified_hash=verify_shard(path,shard)
        save_json(root/'source_receipts'/('rasa_'+shard['path'].replace('/','__')+'.json'), {'path':shard['path'],'bytes':path.stat().st_size,'sha256':verified_hash,'pin':shard})
        used=0
        # Development starts after all prior pilot recordings, including old confirmation.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
            for record in batch.to_pylist():
                if used>=limit: break
                if not valid_speech(record): rejected['metadata']+=1; continue
                if record['language']!=shard['language']: raise ValueError('Language mismatch')
                cid=Path(record['filename']).stem
                raw=record['audio'].get('bytes')
                if not isinstance(raw,bytes): raise ValueError('Missing embedded audio')
                original=hashlib.sha256(raw).hexdigest()
                if cid in excluded_ids or original in excluded_hashes: raise ValueError('Competition source overlap')
                if cid in speech_ids: rejected['repeated_id']+=1; continue
                if shard['split']=='dev' and cid in prior_ids: rejected['previous_development']+=1; continue
                try: x=decode_clean(raw)
                except ValueError: rejected['audio_quality']+=1; continue
                pcm=pcm_hash(x); expected=pcm_wav_hash(x)
                if expected in excluded_hashes: raise ValueError('Competition resampled overlap')
                if expected in seen or pcm in seen_pcm: rejected['duplicate_audio']+=1; continue
                if shard['split']=='dev' and pcm in prior_pcm: rejected['previous_development_pcm']+=1; continue
                out=root/'adapt_data/speech'/(original+'.wav')
                if out.exists() and sha256(out)!=expected: raise ValueError('Prepared speech changed')
                if not out.exists(): write_wav(out,x,len(x))
                if sha256(out)!=expected: raise ValueError('Speech serialization mismatch')
                speech_ids.add(cid);seen.add(expected);seen_pcm.add(pcm);used+=1
                rows.append({'kind':'speech','split':shard['split'],'group':record['language']+'/'+record['gender'],
                    'language':record['language'],'gender':record['gender'],'source_id':cid,'shard':shard['path'],
                    'source_sha256':original,'audio_sha256':expected,'pcm_sha256':pcm,'path':str(out.relative_to(root)),
                    'frames':len(x),'transcript':record['text']})
            if used>=limit: break
        print('Accepted speech in shard:',used,'target:',limit,flush=True)
        save_json(root/'adapt_sources_partial.json',{'status':'incomplete','rows':rows,'rejected':dict(rejected)})
    check_exclusions(rows,parent,holdout)
    counts=Counter((r['split'],r['language']) for r in rows if r['kind']=='speech')
    if any(counts['dev',lang]!=240 for lang in ('Marathi','Tamil')): raise ValueError('Need 240 fresh references per development language')
    if any(counts['train',lang]<150 for lang in ('Hindi','Bengali','Telugu','Kannada')): raise ValueError('Insufficient training coverage')
    save_json(root/'adapt_sources.json',{'rows':rows,'pins_sha256':sha256(PINS),'rejected':dict(rejected),
        'split_policy':pins['split_policy'],'fresh_dev_sources':480,
        'limitations':'Development excludes earlier source IDs/decoded hashes; same two languages and recording corpus remain. Speaker IDs unavailable. Exact ID/hash exclusion cannot rule out re-encoded competition overlap.'})
    prepare_eval(root,rows)
    print('Prepared source counts:',dict(Counter(r['kind']+'/'+r['split'] for r in rows)),flush=True)


def prepare_eval(root, rows):
    root=Path(root); noise=[r for r in rows if r['kind']=='noise' and r['split']=='dev']; result=[]
    for language in ('Marathi','Tamil'):
        speech=sorted([r for r in rows if r['kind']=='speech' and r['split']=='dev' and r['language']==language],key=lambda r:digest(r['source_id']))
        for i,r in enumerate(speech):
            partition,condition_name=evaluation_assignment(i)
            rng=np.random.default_rng(np.random.SeedSequence([101026,i,int(digest(language)[:8],16)]))
            c=read_wav(root/r['path']); bounds=(-15,0) if condition_name=='hard' else (0,20)
            for attempt in range(8):
                try:
                    n=combined_noise(root,noise,len(c),rng);snr=float(rng.uniform(*bounds));level=float(rng.uniform(-30,-12))
                    x,y=mixture(c,n,snr,level,identity=condition_name=='clean'); break
                except ValueError:
                    if attempt==7: raise
            cid='noise_external_'+digest(r['source_id'])[:20]; name=cid+'.wav'
            paths=[root/'external_eval'/kind/name for kind in ('noisy','clean')]
            receipts=[write_wav(p,a,len(a)) for p,a in zip(paths,(x,y))]
            result.append({'clip_id':cid,'filename':name,'subset':'synthetic','frames':len(x),'seconds':len(x)/16000,
                'sha256':receipts[0]['sha256'],'clean_sha256':receipts[1]['sha256'],'transcript':r['transcript'],
                'source_id':r['source_id'],'language':language,'partition':partition,'condition':condition_name,
                'requested_snr_db':None if condition_name=='clean' else snr,'mixture_level_db':level})
    for p in ('select','confirm'):
        if Counter(r['condition'] for r in result if r['partition']==p)!=Counter({'hard':100,'moderate':100,'clean':40}): raise ValueError('External condition balance changed')
    save_json(root/'external_eval/manifest.json',result)
    print('External evaluation: 240 selection + 240 confirmation; noisy/clean metrics separate.',flush=True)


class Pairs:
    def __init__(self,root,arm,frames=64000):
        if arm not in ARMS: raise ValueError('Unknown arm')
        self.root=Path(root); self.arm=arm;self.frames=frames
        doc=json.loads((self.root/'adapt_sources.json').read_text()); check_splits(doc['rows'])
        self.speech=[r for r in doc['rows'] if r['kind']=='speech' and r['split']=='train']
        self.noise=[r for r in doc['rows'] if r['kind']=='noise' and r['split']=='train']
        self.languages=sorted({r['language'] for r in self.speech})
        self.by_language={l:[r for r in self.speech if r['language']==l] for l in self.languages}
        for r in self.speech+self.noise:
            if sha256(self.root/r['path'])!=r['audio_sha256']: raise ValueError('Training audio changed')

    def get(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([102611,index]))
        # Ten-position condition schedule is balanced over all four languages every 40 examples.
        sources=self.by_language[self.languages[(index//10)%len(self.languages)]]
        row=sources[int(rng.integers(len(sources)))]
        for attempt in range(8):
            try:
                clean=crop(read_wav(self.root/row['path']),self.frames,rng)
                noise=combined_noise(self.root,self.noise,self.frames,rng)
                identity,snr=condition(index,self.arm,rng)
                return mixture(clean,noise,snr,float(rng.uniform(-30,-12)),identity)
            except ValueError:
                if attempt==7: raise


if __name__=='__main__':
    try: prepare(Path.cwd(),sys.stdin.read().strip())
    except Exception as error:
        reason=str(error)[:250] if isinstance(error,ValueError) else type(error).__name__
        save_json('noise_data_error.json',{'status':'failed','reason':reason})
        print('Data preparation failed:',reason,flush=True)
        raise SystemExit(1) from None
