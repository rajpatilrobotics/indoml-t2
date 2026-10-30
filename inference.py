#!/usr/bin/env python3
"""Reproduce the final Polar90 waveform recipe, frozen ASR and prediction ZIP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from src.audio import read_wav, write_wav, sha256
from src.data_manifest import save_json
from src.runtime import digest
from experiments.gap_context_v1.geometry import decode
from experiments.gap_refine_v1.geometry import refine
from experiments.gap_polar_v1.geometry import transform

ROOT = Path(__file__).resolve().parent


def pcm_roundtrip(x):
    """Exactly the write_wav/read_wav boundary used between submitted stages."""
    return np.clip(np.rint(np.asarray(x) * 32768), -32768, 32767).astype('<i2').astype(np.float64) / 32768.


def enhance_one(x, native):
    context = pcm_roundtrip(decode(x, native, 'GAPContext20'))
    shifted = pcm_roundtrip(refine(x, context, native, 'GAPShiftMean'))
    return transform(x, shifted, 'GAPPolar90')


def manifest(folder):
    paths = sorted(folder.glob('*.wav'))
    if not paths:
        raise ValueError('Input directory must contain mono PCM16 16 kHz WAV files at its root')
    result = []
    for p in paths:
        x = read_wav(p)
        result.append(dict(clip_id=p.stem, filename=p.name, frames=len(x), seconds=len(x)/16000,
                           sha256=sha256(p), subset='test'))
    if len({r['clip_id'] for r in result}) != len(result):
        raise ValueError('Duplicate clip IDs')
    return result


def verified_model(path):
    import torch
    from experiments.predictive_v1.vendor.tfgridnet import TFGridNet
    spec = json.loads((ROOT/'checkpoints/manifest.json').read_text())
    if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
        raise ValueError('Checkpoint is not the submitted Polar90 model')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['cfg'] != spec['upstream']['config']:
        raise ValueError('Checkpoint architecture mismatch')
    model = TFGridNet(**state['cfg'])
    model.load_state_dict(state['model'], strict=True)
    return model


def worker(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no automatic CPU inference')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    model = verified_model(args.checkpoint).cuda().eval()
    rows = json.loads((args.output/'input_manifest.json').read_text())
    code = {str(p.relative_to(ROOT)): sha256(p) for d in ('src','experiments') for p in (ROOT/d).rglob('*.py')}
    code['inference.py'] = sha256(Path(__file__))
    binding = digest({'checkpoint':sha256(args.checkpoint), 'source':code, 'torch':torch.__version__})
    selected = rows[args.worker_index::args.worker_count]
    def native(block):
        return model(torch.from_numpy(block.copy())[None].cuda(), sr=16000)[0].cpu().numpy()
    for i, r in enumerate(selected, 1):
        original = args.input/r['filename']
        if sha256(original) != r['sha256']:
            raise ValueError('Input changed since manifest creation')
        final = args.output/'audio'/r['filename']
        receipt = args.output/'enhancement_records'/(r['clip_id']+'.json')
        key = digest({'binding':binding, 'input':r})
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['key'] != key or not final.exists() or old['sha256'] != sha256(final):
                raise ValueError('Stale output: choose a new output folder')
            read_wav(final, r['frames'])
        else:
            seed = 2026 ^ int(r['sha256'][:8], 16)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                y = enhance_one(read_wav(original, r['frames']), native)
            save_json(receipt, {'key':key, **write_wav(final, y, r['frames'])})
        if i == 1 or i % 25 == 0 or i == len(selected):
            print(f'Enhancement worker {args.worker_index}: {i}/{len(selected)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='Directory containing original test WAVs')
    p.add_argument('--output', type=Path, default=Path('runs/polar90'))
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/polar90.pt')
    p.add_argument('--gpus', default='0', help='CUDA GPU indices for enhancement, e.g. 0,1')
    p.add_argument('--enhance-only', action='store_true', help='Skip ASR and ZIP; audio only')
    p.add_argument('--official', action='store_true', help='Require exactly 3178 Phase 2 clips')
    p.add_argument('--verify-checkpoint', action='store_true', help='Hash and strict CPU load only, no inference')
    p.add_argument('--worker-index', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--worker-count', type=int, default=1, help=argparse.SUPPRESS)
    a = p.parse_args(); a.checkpoint = a.checkpoint.resolve()
    if a.verify_checkpoint:
        model = verified_model(a.checkpoint)
        print(json.dumps({'status':'PASS', 'sha256':sha256(a.checkpoint), 'parameters':sum(p.numel() for p in model.parameters())}))
        return
    if a.input is None:
        p.error('--input is required')
    a.input = a.input.resolve(); a.output = a.output.resolve()
    if a.input == a.output or a.input == a.output/'audio':
        p.error('Input and output must be separate')
    if a.worker_index is not None:
        worker(a); return
    rows = manifest(a.input)
    if a.official and len(rows) != 3178:
        raise ValueError('Official Phase 2 requires all 3178 input WAVs')
    a.output.mkdir(parents=True, exist_ok=True)
    mp = a.output/'input_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != rows:
        raise ValueError('Existing output belongs to different inputs')
    save_json(mp, rows)
    (a.output/'input_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    gpus = a.gpus.split(',')
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        p.error('--gpus must be distinct integer indices')
    processes = []
    for i, gpu in enumerate(gpus):
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--input', str(a.input), '--output', str(a.output),
               '--checkpoint', str(a.checkpoint), '--worker-index', str(i), '--worker-count', str(len(gpus))]
        processes.append(subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES':gpu}))
    codes = [proc.wait() for proc in processes]
    if any(codes):
        raise RuntimeError(f'Enhancement failed: {codes}; valid completed clips are resumable')
    if a.enhance_only:
        print(f'Enhanced {len(rows)} WAVs: {a.output / "audio"}'); return
    # The same mandated API runs on the final serialized files; no text corrections.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Set HF_TOKEN in the environment, with accepted SraVaani access')
    from experiments.gap_token_check_v1.snapshot import prepare_snapshot
    asr = json.loads((ROOT/'configs/asr.json').read_text())
    save_json(a.output/'configs/phase2_gap_adapt.json', {'validated_asr_config':asr})
    prepare_snapshot(a.output, token)
    from src.transcribe import transcribe
    from src.package_submission import package
    transcribe(a.output/'input_manifest.jsonl', a.output/'audio', a.output/'asr', token=token)
    target = a.output/'prediction.zip'
    if target.exists():
        raise FileExistsError('prediction.zip already exists; preserve it or choose a new output folder')
    audit = package(a.output/'audio', a.output/'asr/transcripts.jsonl', a.output/'asr/asr_provenance.json', rows, target, official=a.official)
    summary = {k:v for k,v in audit.items() if k != 'wav_sha256'}
    save_json(a.output/'RUN_COMPLETED.json', {'status':'complete', 'candidate':'GAPPolar90', **summary})
    print(json.dumps({'download':str(target), **summary}, indent=2))

if __name__ == '__main__':
    main()
