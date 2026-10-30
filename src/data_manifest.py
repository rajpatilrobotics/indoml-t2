"""Independently audit original ZIPs and build deterministic private manifests."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
from .audio import read_wav, sha256
from .metrics import normalize, si_sdr


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['clip_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate manifest IDs')
    for row in rows:
        if row['filename'] != row['clip_id'] + '.wav' or Path(row['filename']).name != row['filename']:
            raise ValueError('Unsafe or inconsistent manifest filename')
    return rows


def write_manifest(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def audit(validation_zip, test_zip, expected_path, output_dir):
    expected = json.loads(Path(expected_path).read_text())
    output_dir = Path(output_dir)
    all_rows, summary = [], {'archives': {}, 'events_outside_audio': []}
    original_scores = []
    for kind, archive in [('validation', Path(validation_zip)), ('test', Path(test_zip))]:
        exp = expected['archives'][archive.name]
        digest = sha256(archive)
        if digest != exp['sha256'] or archive.stat().st_size != exp['file_bytes']:
            raise ValueError(f'Archive integrity mismatch: {archive.name}')
        rows = []
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP entry')
            wav_names = [n for n in names if n.endswith('.wav') and '__MACOSX' not in PurePosixPath(n).parts and not PurePosixPath(n).name.startswith('._')]
            if len(wav_names) != exp['wav_count']:
                raise ValueError('Wrong WAV count')
            metadata = json.loads(z.read('validation/validationMetadata.json')) if kind == 'validation' else []
            by_name = {m['segmentFileName']: m for m in metadata}
            if len(by_name) != len(metadata):
                raise ValueError('Duplicate metadata ID')
            audio_data = {}
            # Reading the full entry checks ZIP CRC as well as the WAV payload length.
            for member in sorted(wav_names):
                raw = z.read(member)
                wav = read_wav(raw)
                audio_data[member] = {'frames': len(wav), 'sha256': hashlib.sha256(raw).hexdigest()}
            for member in sorted(wav_names):
                if '/syntheticCleanRefAudio/' in member:
                    continue
                name = PurePosixPath(member).name
                info = audio_data[member]
                m = by_name[name] if kind == 'validation' else {}
                subset = ('synthetic' if m['syntheticData'] else 'natural') if m else 'test'
                if m:
                    expected_dir = 'syntheticNoiseAudio' if m['syntheticData'] else 'naturalNoisyAudio'
                    if member != f'validation/{expected_dir}/{name}':
                        raise ValueError('Metadata/audio source mismatch')
                row = {'clip_id': name[:-4], 'filename': name, 'archive': archive.name,
                       'member': member, 'subset': subset, 'frames': info['frames'],
                       'sample_rate': 16000, 'seconds': info['frames']/16000, 'sha256': info['sha256']}
                if m:
                    row.update(language=m['language'], transcript=m['transcript'],
                               reference_words=len(normalize(m['transcript']).split()),
                               event_categories=m['NoiseCategory'], annotation_quality=m['annotationQuality'])
                    row['events'] = []
                    for event in m['NoiseSubCategoryTimeStamp']:
                        start, end = float(event['start']), float(event['end'])
                        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                            raise ValueError('Invalid event annotation')
                        row['events'].append(dict(event, start=start, end=end))
                        if end > row['seconds'] + .02:
                            summary['events_outside_audio'].append({'clip_id':row['clip_id'], 'end':end, 'duration':row['seconds']})
                    if subset == 'synthetic':
                        clean_member = f'validation/syntheticCleanRefAudio/{name}'
                        clean = audio_data[clean_member]
                        if clean['frames'] != info['frames']:
                            raise ValueError('Pair length mismatch')
                        row.update(clean_member=clean_member, clean_sha256=clean['sha256'])
                        score = si_sdr(read_wav(z.read(clean_member)), read_wav(z.read(member)))
                        row['original_si_sdr_db'] = score
                        original_scores.append(score)
                rows.append(row)
            if kind == 'validation' and {r['filename'] for r in rows} != set(by_name):
                raise ValueError('Metadata correspondence mismatch')
        write_manifest(output_dir / f'{kind}.jsonl', rows)
        all_rows.extend(rows)
        summary['archives'][archive.name] = {'sha256': digest, 'bytes': archive.stat().st_size, 'wav_count':len(wav_names), 'input_count':len(rows), 'input_hours':sum(r['seconds'] for r in rows)/3600, 'all_audio_crc_verified':True}
    summary['original_noisy_si_sdr_mean_db'] = float(np.mean(original_scores))
    if not np.isclose(summary['original_noisy_si_sdr_mean_db'], expected['validation']['original_noisy_si_sdr_mean_db'], atol=1e-9, rtol=0):
        raise ValueError('SI-SDR reproduction failed')
    summary['subsets'] = {s:{'clips':sum(r['subset']==s for r in all_rows), 'reference_words':sum(r.get('reference_words',0) for r in all_rows if r['subset']==s)} for s in ['synthetic','natural','test']}
    # Fixed language-diverse small smoke set; test examples are inference-only.
    smoke = []
    for subset in ['synthetic', 'natural']:
        group = sorted((r for r in all_rows if r['subset']==subset), key=lambda r: hashlib.sha256(r['clip_id'].encode()).hexdigest())
        chosen, languages = [], set()
        for r in group:
            if r['language'] not in languages:
                chosen.append(r); languages.add(r['language'])
            if len(chosen) == 4:
                break
        smoke.extend(chosen)
    tests = sorted((r for r in all_rows if r['subset']=='test'), key=lambda r:(r['frames'],r['clip_id']))
    smoke.extend([tests[len(tests)//2], tests[-1]])
    write_manifest(output_dir/'smoke10.jsonl', smoke)
    summary['smoke'] = {'clips':len(smoke),'seconds':sum(r['seconds'] for r in smoke),'longest_seconds':max(r['seconds'] for r in smoke)}
    save_json(output_dir/'audit.json', summary)
    return summary


def extract(rows, archives_dir, output_dir):
    output_dir = Path(output_dir)
    for row in rows:
        with zipfile.ZipFile(Path(archives_dir)/row['archive']) as z:
            members = [('audio',row['member'],row['sha256'])]
            if 'clean_member' in row:
                members.append(('clean',row['clean_member'],row['clean_sha256']))
            for folder, member, expected in members:
                raw=z.read(member)
                if hashlib.sha256(raw).hexdigest()!=expected:
                    raise ValueError('Input hash mismatch')
                path=output_dir/folder/row['filename']
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists() and sha256(path)!=expected:
                    raise ValueError(f'Refusing to overwrite changed input: {path}')
                if not path.exists(): path.write_bytes(raw)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archives-dir',type=Path,required=True)
    parser.add_argument('--expected',default='configs/expected_data_audit.json')
    parser.add_argument('--output',type=Path,default=Path('data/manifests'))
    args=parser.parse_args()
    result=audit(args.archives_dir/'validation.zip',args.archives_dir/'input_data_track-2_phase-2.zip',args.expected,args.output)
    extract(read_manifest(args.output/'smoke10.jsonl'),args.archives_dir,Path('data/smoke10'))
    print(json.dumps(result,indent=2))
