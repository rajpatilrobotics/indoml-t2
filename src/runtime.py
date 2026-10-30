"""Small run records and content-addressed caches, without environment secrets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .data_manifest import save_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def environment():
    return {'python':sys.version,'packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions() if p.metadata['Name']}}


def source_digest():
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This model runner requires a Kaggle CUDA GPU; no CPU fallback')
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark=False
    return torch


def cached_record(path,key,audio_hash=None):
    if not Path(path).exists(): return None
    record=json.loads(Path(path).read_text())
    if record.get('cache_key') != key or record.get('status')!='success': return None
    if audio_hash is not None and record.get('audio_sha256')!=audio_hash:return None
    return record
