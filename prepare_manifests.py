#!/usr/bin/env python3
"""Create local ID/hash exclusion manifests from organizer-supplied WAV folders."""
import argparse,json
from pathlib import Path
from inference import manifest
from src.audio import read_wav,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--validation',type=Path,required=True)
p.add_argument('--validation-clean',type=Path,required=True)
p.add_argument('--test',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for split,folder in [('validation',a.validation),('test',a.test)]:
 rows=manifest(folder)
 for r in rows:
  r['subset']=split
  if split=='validation':
   clean=a.validation_clean/r['filename']
   if clean.exists():
    read_wav(clean,r['frames']);r['clean_sha256']=sha256(clean)
 (a.output/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('Wrote exclusion manifests; no speech or transcript content copied.')
