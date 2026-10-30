"""Strict PCM16 audio boundary. No resampling or silent length repair."""
from pathlib import Path
import hashlib
import io
import wave
import numpy as np


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_wav(source, expected_frames=None):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    with wave.open(source, 'rb') as wav:
        spec = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype())
        if spec != (16000, 1, 2, 'NONE'):
            raise ValueError(f'Expected mono PCM16 16 kHz, got {spec}')
        frames = wav.getnframes()
        if frames <= 0 or (expected_frames is not None and frames != expected_frames):
            raise ValueError(f'Invalid frame count {frames}; expected {expected_frames}')
        raw = wav.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError('Truncated WAV payload')
    return np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0


def write_wav(path, audio, expected_frames):
    audio = np.asarray(audio)
    if audio.ndim != 1 or len(audio) != expected_frames or not np.isfinite(audio).all():
        raise ValueError('Output must be finite, mono, and exactly the original length')
    # One explicit quantizer for all clips. Report clipping rather than hiding it.
    clipped = int(np.count_nonzero((audio < -1) | (audio > 32767 / 32768)))
    pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype('<i2')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.wav')
    with wave.open(str(temporary), 'wb') as wav:
        wav.setparams((1, 2, 16000, expected_frames, 'NONE', 'not compressed'))
        wav.writeframes(pcm.tobytes())
    read_wav(temporary, expected_frames)
    temporary.replace(path)
    return {'sha256': sha256(path), 'frames': expected_frames, 'clipped_samples': clipped}
