"""Pinned external sources, source-disjoint splits, deterministic paired mixtures."""
import hashlib
import io
import json
import re
import sys
import urllib.request
import wave
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from src.audio import read_wav, write_wav, sha256
from src.data_manifest import save_json, read_manifest
from src.runtime import digest

RATE = 16000
PINS = Path(__file__).with_name('data_pins.json')


def crop(x, size, rng):
    if len(x) < size:
        raise ValueError('Source is shorter than the fixed crop')
    start = int(rng.integers(0, len(x) - size + 1))
    return np.asarray(x[start:start + size], dtype=np.float64).copy()


def mixture(clean, noise, snr, level_db, identity=False):
    """Common scaling preserves the exact target relation and requested SNR."""
    c = np.asarray(clean, dtype=np.float64).copy()
    n = np.asarray(noise, dtype=np.float64).copy()
    if c.ndim != 1 or c.shape != n.shape or not np.isfinite(c).all() or not np.isfinite(n).all():
        raise ValueError('Invalid paired waveforms')
    c -= c.mean(); n -= n.mean()
    cr = float(np.sqrt(np.mean(c * c))); nr = float(np.sqrt(np.mean(n * n)))
    if min(cr, nr) < 1e-6:
        raise ValueError('Silent crop; never fabricate a training target')
    n *= cr / nr * 10 ** (-snr / 20)
    if identity:
        n.fill(0)
    x = c + n
    gain = 10 ** (level_db / 20) / max(float(np.sqrt(np.mean(x * x))), 1e-12)
    gain = min(gain, .95 / max(float(np.max(np.abs(x))), float(np.max(np.abs(c))), 1e-12))
    return (x * gain).astype(np.float32), (c * gain).astype(np.float32)


def check_splits(rows):
    for kind in ('speech', 'noise'):
        groups = {}; hashes = {}
        for split in ('train', 'dev'):
            selected = [r for r in rows if r['kind'] == kind and r['split'] == split]
            if not selected:
                raise ValueError('Empty source split')
            groups[split] = {r['group'] for r in selected}
            hashes[split] = {r['audio_sha256'] for r in selected}
        if groups['train'] & groups['dev'] or hashes['train'] & hashes['dev']:
            raise ValueError('Training/development source overlap')
        pcm = {split: {r['pcm_sha256'] for r in rows if r['kind'] == kind and r['split'] == split and 'pcm_sha256' in r}
               for split in ('train', 'dev')}
        if pcm['train'] & pcm['dev']:
            raise ValueError('Training/development decoded PCM overlap')


def pcm_hash(audio):
    pcm = np.clip(np.rint(np.asarray(audio) * 32768), -32768, 32767).astype('<i2')
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def validate_noise_pins(noise):
    """Reject duplicates in the manifest before any large data download."""
    files = noise['files']
    if not files:
        raise ValueError('Noise source manifest is empty')
    for field in ('path', 'git_blob_sha1', 'sha256', 'pcm_sha256'):
        seen = {}
        for f in files:
            value = f.get(field)
            if not value:
                raise ValueError('Noise source pin missing: ' + field)
            if value in seen:
                raise ValueError('Duplicate noise ' + field + ': ' + seen[value] + ' and ' + f['path'])
            seen[value] = f['path']
    for split in ('train', 'dev'):
        if not any(f['split'] == split for f in files):
            raise ValueError('Missing noise split: ' + split)
    if {f['category'] for f in files if f['split'] == 'train'} & {f['category'] for f in files if f['split'] == 'dev'}:
        raise ValueError('Noise categories overlap between train and development')


def prepare_noise(root, noise, excluded_hashes):
    root = Path(root); validate_noise_pins(noise); rows = []
    save_json(root / 'adapt_noise_verified.json', {'status': 'checking', 'sources': 0})
    for i, f in enumerate(noise['files'], 1):
        print('Checking noise:', f['path'], f'({i}/{len(noise["files"])})', flush=True)
        save_json(root / 'adapt_data_stage.json', {'stage': 'noise', 'source': f['path'], 'index': i, 'total': len(noise['files'])})
        out = root / 'adapt_data/noise' / Path(f['path']).name
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


def valid_speech(record):
    name = record.get('filename', '')
    language = record.get('language', '')
    gender = record.get('gender', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_ -]+(?:\.wav)?', name):
        return False
    text = str(record.get('text', ''))
    if not text.strip() or '[' in text or ']' in text or '<' in text:
        return False
    return bool(language and gender and 4 <= float(record['duration']) <= 20)


def decode_clean(raw):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(raw), dtype='float64')
    if sr != 48000 or x.ndim != 1 or not 4 * sr <= len(x) <= 20 * sr or not np.isfinite(x).all():
        raise ValueError('Unexpected Rasa audio format')
    if np.mean(np.abs(x) >= .999) > .001 or np.std(x) < .003:
        raise ValueError('Clipped or near-silent target')
    return resample_poly(x, 1, 3, window=('kaiser', 8.6))


def pcm_wav_hash(x):
    buf = io.BytesIO()
    pcm = np.clip(np.rint(np.asarray(x) * 32768), -32768, 32767).astype('<i2')
    with wave.open(buf, 'wb') as w:
        w.setparams((1, 2, RATE, len(x), 'NONE', 'not compressed')); w.writeframes(pcm.tobytes())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def prepare(root, token):
    root = Path(root); directory = root / 'adapt_data'
    pins = json.loads(PINS.read_text()); validate_noise_pins(pins['noise'])
    # Isolate Arrow from the frozen recognizer's package environment.
    sys.path.insert(0, '/kaggle/temp/adapt_arrow')
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rejected = Counter()
    # This manifest records only IDs/hashes for exclusion, not training labels.
    holdout = read_manifest(root / 'data/manifests/validation.jsonl') + read_manifest(root / 'data/manifests/test.jsonl')
    excluded_ids = {r['clip_id'] for r in holdout}
    excluded_hashes = {r['sha256'] for r in holdout} | {r['clean_sha256'] for r in holdout if 'clean_sha256' in r}
    rows = prepare_noise(root, pins['noise'], excluded_hashes)
    seen = {r['audio_sha256'] for r in rows}; seen_pcm = {r['pcm_sha256'] for r in rows}
    for shard in pins['rasa']['shards']:
        print('Downloading/checking Rasa:', shard['path'], flush=True)
        save_json(root / 'adapt_data_stage.json', {'stage': 'speech', 'source': shard['path']})
        path = Path(hf_hub_download(pins['rasa']['repo'], shard['path'], repo_type='dataset', revision=pins['rasa']['revision'], token=token))
        if path.stat().st_size != shard['bytes'] or sha256(path) != shard['sha256']:
            raise ValueError('Pinned Rasa shard mismatch')
        used = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
            for record in batch.to_pylist():
                if used >= shard['limit']:
                    break
                if not valid_speech(record):
                    rejected['metadata'] += 1; continue
                if record['language'] != shard['language']:
                    raise ValueError('Rasa language metadata mismatch')
                raw = record['audio'].get('bytes')
                if not isinstance(raw, bytes):
                    raise ValueError('Expected embedded Parquet audio bytes')
                original_hash = hashlib.sha256(raw).hexdigest()
                cid = Path(record['filename']).stem
                if cid in excluded_ids or original_hash in excluded_hashes:
                    raise ValueError('External speech overlaps competition data')
                try:
                    x = decode_clean(raw)
                except ValueError:
                    rejected['audio_quality'] += 1; continue
                path_out = directory / 'speech' / (hashlib.sha256(raw).hexdigest() + '.wav')
                expected_hash = pcm_wav_hash(x)
                if path_out.exists() and sha256(path_out) != expected_hash:
                    raise ValueError('Previously prepared speech changed')
                receipt = write_wav(path_out, x, len(x)) if not path_out.exists() else {'sha256': sha256(path_out)}
                if receipt['sha256'] != expected_hash: raise ValueError('Unexpected target quantization')
                read_wav(path_out, len(x))
                if receipt['sha256'] in excluded_hashes:
                    raise ValueError('Resampled speech overlaps competition data')
                decoded_hash = pcm_hash(x)
                if receipt['sha256'] in seen or decoded_hash in seen_pcm:
                    rejected['duplicate'] += 1; continue
                seen.add(receipt['sha256']); seen_pcm.add(decoded_hash); used += 1
                rows.append({'kind': 'speech', 'split': shard['split'], 'group': record['language'] + '/' + record['gender'],
                             'language': record['language'], 'gender': record['gender'], 'source_id': cid,
                             'shard': shard['path'], 'source_sha256': original_hash, 'audio_sha256': receipt['sha256'], 'pcm_sha256': decoded_hash,
                             'path': str(path_out.relative_to(root)), 'frames': len(x)})
            if used >= shard['limit']:
                break
        print('Accepted speech from shard:', used, flush=True)
        save_json(root / 'adapt_data_progress.json', {'speech_sources': sum(r['kind'] == 'speech' for r in rows), 'noise_sources': sum(r['kind'] == 'noise' for r in rows), 'rejected': dict(rejected)})
        save_json(root / 'adapt_sources_partial.json', {'status': 'incomplete_preparation', 'pins_sha256': sha256(PINS), 'rows': rows})
    del token
    check_splits(rows)
    counts = Counter((r['split'], r.get('language', 'noise')) for r in rows)
    if any(counts[('train', lang)] < 150 for lang in ('Hindi', 'Bengali', 'Telugu', 'Kannada')) or any(counts[('dev', lang)] < 40 for lang in ('Marathi', 'Tamil')):
        raise ValueError('Insufficient quality-filtered language coverage')
    document = {'rows': rows, 'pins_sha256': sha256(PINS), 'split_policy': pins['split_policy'], 'rejected': dict(rejected),
                'exclusions': 'Exact source IDs and encoded/resampled WAV hashes checked against official manifests. Re-encoded or speaker overlap with competition data cannot be ruled out.',
                'quality_limits': 'Source labels and numeric checks are not a perceptual certification. Rasa studio speech differs from natural Vaani speech; pilot benefit is unknown.'}
    save_json(root / 'adapt_sources.json', document)
    save_json(root / 'adapt_data_stage.json', {'stage': 'complete', 'sources': len(rows)})
    print('Prepared sources:', dict(Counter(r['kind'] + '/' + r['split'] for r in rows)), flush=True)


class Pairs:
    def __init__(self, root, split, frames=64000):
        self.root = Path(root); self.frames = frames
        self.document = json.loads((self.root / 'adapt_sources.json').read_text())
        check_splits(self.document['rows'])
        self.speech = [r for r in self.document['rows'] if r['kind'] == 'speech' and r['split'] == split]
        self.noise = [r for r in self.document['rows'] if r['kind'] == 'noise' and r['split'] == split]
        self.languages = sorted({r['language'] for r in self.speech})
        self.by_language = {lang: [r for r in self.speech if r['language'] == lang] for lang in self.languages}
        self.seed = 31009 if split == 'train' else 71009
        for r in self.speech + self.noise:
            if sha256(self.root / r['path']) != r['audio_sha256']:
                raise ValueError('Prepared source changed')

    def get(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        sources = self.by_language[self.languages[index % len(self.languages)]]
        c = sources[int(rng.integers(len(sources)))]; n = self.noise[int(rng.integers(len(self.noise)))]
        # Retry crops inside the same disjoint sources, not by changing targets.
        for _ in range(8):
            clean = crop(read_wav(self.root / c['path']), self.frames, rng)
            noise = crop(read_wav(self.root / n['path']), self.frames, rng)
            try:
                return mixture(clean, noise, float(rng.uniform(-5, 20)), float(rng.uniform(-30, -12)), identity=(index % 5 == 0))
            except ValueError:
                pass
        raise ValueError('No non-silent crop after eight deterministic attempts')


if __name__ == '__main__':
    try:
        prepare(Path.cwd(), sys.stdin.read().strip())
    except Exception as error:
        reason = str(error)[:250] if type(error) is ValueError else None
        save_json('adapt_data_error.json', {'status': 'failed', 'error_type': type(error).__name__,
                                          'data_check': reason,
                                          'hint': 'Inspect adapt_data_stage.json and this named data check. Authentication is only implicated by an access error; no authenticated URLs or tokens are logged.'})
        print('External data preparation failed:', type(error).__name__, '|', reason or 'See the saved stage/error records.', flush=True)
        raise SystemExit(1) from None
