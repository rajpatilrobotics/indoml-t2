import numpy as np
import pytest
from inference import enhance_one,pcm_roundtrip
from src.audio import write_wav,read_wav
from experiments.gap_context_v1.geometry import decode
from experiments.gap_refine_v1.geometry import refine
from experiments.gap_polar_v1.geometry import transform,stft,istft
from experiments.gap_balance_v1.objective import combine_gradients
import torch

@pytest.mark.parametrize('n',[1,129,1024,320001,650119])
def test_context_keeps_every_sample(n):
    x=np.random.default_rng(1).normal(0,.1,n).astype(np.float32)
    np.testing.assert_array_equal(decode(x,lambda v:v,'GAPContext20'),x)

@pytest.mark.parametrize('n',[1025,320117])
def test_consolidation_matches_original_disk_stage_boundaries(tmp_path,n):
    x=np.random.default_rng(4).normal(0,.15,n)
    # Nonlinear and position-sensitive stub exposes shift/context mistakes.
    def native(v):return np.tanh(.8*v+np.arange(len(v),dtype=np.float32)%256*1e-5).astype(np.float32)
    first=tmp_path/'context.wav';second=tmp_path/'shift.wav'
    write_wav(first,decode(x,native,'GAPContext20'),n)
    write_wav(second,refine(x,read_wav(first),native,'GAPShiftMean'),n)
    expected=transform(x,read_wav(second),'GAPPolar90')
    np.testing.assert_array_equal(enhance_one(x,native),expected)

@pytest.mark.parametrize('n',[1,127,512,16003])
def test_polar_roundtrip(n):
    x=np.random.default_rng(8).normal(0,.1,n)
    np.testing.assert_allclose(istft(stft(x),n),x,atol=1e-14)
    np.testing.assert_allclose(transform(x,.2*x,'GAPPolar90'),.2**.9*x,atol=1e-14)

def test_pcm_matches_writer_and_clipping(tmp_path):
    x=np.array([-2,-1,-.12345,0,.99999,1,2]);p=tmp_path/'x.wav'
    write_wav(p,x,len(x));np.testing.assert_array_equal(pcm_roundtrip(x),read_wav(p))

def test_conflicting_recognition_gradient_respects_cap():
    rec=[torch.tensor([2.,0.])];kd=[torch.tensor([-5.,10.])]
    result,details=combine_gradients(rec,kd)
    assert details['conflict_projected']
    assert details['effective_recognition_ratio']<=.5+1e-6
    assert torch.dot(rec[0],result[0]-rec[0])>=0

def test_submission_packaging_and_tamper_rejection(tmp_path):
    import hashlib,json
    from src.audio import sha256
    from src.package_submission import package,validate_zip
    from src.transcribe import ASR_REPO,ASR_REVISION,DECODER
    from src.data_manifest import save_json
    audio=tmp_path/'audio';audio.mkdir()
    rows=[];clips={};lines=[]
    for i in range(2):
        cid=f'test_fixture_{i}';name=cid+'.wav';n=2048+i
        receipt=write_wav(audio/name,np.random.default_rng(i).normal(0,.05,n),n)
        text='synthetic test transcript'
        rows.append({'clip_id':cid,'filename':name,'frames':n,'subset':'test'})
        clips[cid]={'audio_sha256':receipt['sha256'],'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'status':'success'}
        lines.append(json.dumps({'clip_id':cid,'text':text})+'\n')
    transcript=tmp_path/'transcripts.jsonl';transcript.write_text(''.join(lines))
    prov=tmp_path/'provenance.json'
    save_json(prov,{'clips':clips,'asr_repo':ASR_REPO,'asr_revision':ASR_REVISION,'decoder':DECODER,'transcripts_sha256':sha256(transcript)})
    z=tmp_path/'prediction.zip';result=package(audio,transcript,prov,rows,z)
    assert result['wav_count']==2 and validate_zip(z,rows)['transcript_count']==2
    write_wav(audio/rows[0]['filename'],np.zeros(rows[0]['frames']),rows[0]['frames'])
    with pytest.raises(ValueError,match='Stale audio'):
        package(audio,transcript,prov,rows,tmp_path/'tampered.zip')
