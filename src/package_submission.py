"""Fail-closed prediction packaging; provenance remains outside the prediction ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
from .audio import read_wav, sha256
from .data_manifest import read_manifest, save_json


def parse_transcripts(raw, expected_ids):
    records = {}
    for i, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            raise ValueError(f'Blank JSONL line {i}')
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != {'clip_id', 'text'}:
            raise ValueError('Transcript fields must be exactly clip_id and text')
        cid, text = record['clip_id'], record['text']
        if not isinstance(cid, str) or not isinstance(text, str) or cid in records:
            raise ValueError('Duplicate ID or invalid transcript type')
        records[cid] = text
    if set(records) != set(expected_ids):
        raise ValueError('Transcript IDs do not exactly match the manifest')
    return records


def validate_zip(path, rows):
    expected = {r['filename']:r for r in rows}
    if len(expected) != len(rows) or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest ID')
    if any(Path(name).name != name or name != row['clip_id']+'.wav' for name,row in expected.items()):
        raise ValueError('Unsafe manifest filename')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | {'transcripts.jsonl'}:
            raise ValueError('ZIP must contain exactly manifest WAVs and transcripts.jsonl at root')
        transcripts = parse_transcripts(z.read('transcripts.jsonl'), {r['clip_id'] for r in rows})
        hashes = {}
        for name, row in expected.items():
            raw = z.read(name)
            read_wav(raw, row['frames'])
            hashes[name] = hashlib.sha256(raw).hexdigest()
        transcript_hash = hashlib.sha256(z.read('transcripts.jsonl')).hexdigest()
    return {'wav_count':len(rows), 'transcript_count':len(transcripts), 'wav_sha256':hashes,
            'transcripts_sha256':transcript_hash, 'zip_sha256':sha256(path)}


def package(audio_dir, transcript_path, provenance_path, rows, output, official=False):
    if official and (len(rows)!=3178 or any(r['subset']!='test' for r in rows)):
        raise ValueError('Official Phase 2 submission requires all 3,178 test rows')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a prediction ZIP')
    expected_ids={r['clip_id'] for r in rows}
    raw_text=Path(transcript_path).read_bytes()
    texts=parse_transcripts(raw_text,expected_ids)
    provenance=json.loads(Path(provenance_path).read_text())
    if set(provenance['clips'])!=expected_ids:
        raise ValueError('Incomplete ASR provenance')
    if provenance['asr_repo']!='ARTPARK-IISc/SraVaani-1.0' or provenance['asr_revision']!='f5dd5358325a5208775b91dad98918e079ea2b27':
        raise ValueError('Unapproved ASR provenance')
    if provenance.get('decoder')!='official model.transcribe defaults, return_hypotheses=True':
        raise ValueError('Unexpected decoder configuration')
    if provenance.get('transcripts_sha256')!=hashlib.sha256(raw_text).hexdigest():
        raise ValueError('Transcript file differs from ASR provenance')
    for row in rows:
        path=Path(audio_dir)/row['filename']
        record=provenance['clips'][row['clip_id']]
        if record['audio_sha256']!=sha256(path) or record['text_sha256']!=hashlib.sha256(texts[row['clip_id']].encode()).hexdigest() or record['status']!='success':
            raise ValueError(f'Stale audio/transcript provenance: {row["clip_id"]}')
        read_wav(path,row['frames'])
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial.zip')
    # Stable ZIP metadata makes packaging byte-reproducible.
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for row in sorted(rows,key=lambda r:r['filename']):
            z.writestr(zipfile.ZipInfo(row['filename'],date_time=(2026,1,1,0,0,0)), (Path(audio_dir)/row['filename']).read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
        z.writestr(zipfile.ZipInfo('transcripts.jsonl',date_time=(2026,1,1,0,0,0)),raw_text,compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
    report=validate_zip(temporary,rows)
    temporary.replace(output)
    report['official_phase2']=official
    save_json(output.with_suffix('.audit.json'),report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--audio-dir',required=True)
    p.add_argument('--transcripts',required=True)
    p.add_argument('--provenance',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--official',action='store_true')
    a=p.parse_args()
    report=package(a.audio_dir,a.transcripts,a.provenance,read_manifest(a.manifest),a.output,a.official)
    print(json.dumps({k:v for k,v in report.items() if k!='wav_sha256'},indent=2))
