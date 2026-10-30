"""Fetch every pinned ASR artifact explicitly; never trust a partial snapshot."""
import json
import sys
import time
from pathlib import Path
from src.audio import sha256
from src.data_manifest import save_json

SUFFIXES={'.ts','.pt','.model','.json','.py'}


def differences(folder, expected):
    found={p.name:sha256(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix in SUFFIXES}
    return found, {'missing':sorted(set(expected)-set(found)), 'unexpected':sorted(set(found)-set(expected)),
                   'mismatched':{name:{'expected':expected[name],'actual':found[name]} for name in sorted(set(found)&set(expected)) if found[name]!=expected[name]}}


def prepare_snapshot(root, token, download=None, pause=time.sleep):
    if download is None:
        from huggingface_hub import hf_hub_download
        download=hf_hub_download
    root=Path(root)
    spec=json.loads((root/'configs/phase2_gap_adapt.json').read_text())['validated_asr_config']
    expected=spec['artifact_sha256']
    audit={'status':'fetching','repo':spec['asr_repo'],'revision':spec['asr_revision'],
           'expected_artifact_sha256':expected,'files':{},'retry_limit_per_file':3}
    audit_path=root/'teacher_download_audit.json'
    save_json(audit_path,audit)
    folder=None
    for index,(name,pin) in enumerate(sorted(expected.items()),1):
        if Path(name).name!=name:raise ValueError('Unexpected nested ASR artifact pin')
        attempts=[];audit['files'][name]=attempts;force=False
        for attempt in range(1,4):
            record={'attempt':attempt,'force_download':force}
            try:
                path=Path(download(repo_id=spec['asr_repo'],filename=name,revision=spec['asr_revision'],
                                   token=token,local_files_only=False,force_download=force))
                # Keep the snapshot symlink path, not its resolved blobs directory.
                if path.name!=name or path.parent.name!=spec['asr_revision']:
                    raise ValueError('Unexpected ASR download path or revision')
                if folder is not None and path.parent!=folder:raise ValueError('Mixed ASR snapshot directories')
                folder=path.parent
                actual=sha256(path);record.update(sha256=actual,bytes=path.stat().st_size)
                if actual!=pin:
                    force=True
                    raise ValueError('ASR artifact hash mismatch')
                record['status']='verified';attempts.append(record);save_json(audit_path,audit)
                print(f'Frozen ASR: {index}/{len(expected)} verified: {name}',flush=True)
                break
            except Exception as error:
                record.update(status='failed',error_type=type(error).__name__)
                response=getattr(error,'response',None)
                if response is not None:record['http_status']=getattr(response,'status_code',None)
                # Exception messages may contain signed URLs or credentials; record types only.
                attempts.append(record)
                if folder is not None and folder.is_dir():
                    _,audit['snapshot_difference']=differences(folder,expected)
                audit['status']='failed' if attempt==3 else 'retrying'
                save_json(audit_path,audit)
                if attempt==3:
                    raise RuntimeError(f'Frozen ASR download/verification failed for {name} after 3 attempts. See teacher_download_audit.json.') from None
                print(f'Frozen ASR: retrying {name}; {type(error).__name__} (attempt {attempt}/3)',flush=True)
                pause(2*attempt)
        audit['status']='fetching'
    found,delta=differences(folder,expected);audit['snapshot_difference']=delta
    if found!=expected:
        audit['status']='failed';save_json(audit_path,audit)
        raise ValueError('Frozen ASR snapshot contains missing, changed or unexpected files. See teacher_download_audit.json.')
    receipt={'path':str(folder),'artifact_sha256':found,'revision':spec['asr_revision']}
    save_json(root/'teacher_snapshot.json',receipt)
    audit['status']='PASS';save_json(audit_path,audit)
    print('All pinned frozen ASR files verified. Recognition model and decoding are unchanged.',flush=True)
    return folder


if __name__=='__main__':
    prepare_snapshot(Path.cwd(),sys.stdin.read().strip())
