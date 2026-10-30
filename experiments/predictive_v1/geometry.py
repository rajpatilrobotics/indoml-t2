"""Waveform-only, reference-independent context policy for both challengers."""
import numpy as np
from scipy.signal import resample_poly

WINDOW = 8 * 16000
STRIDE = 6 * 16000
MARGIN = 16000
HOP = 256  # GAP's hop; also maps to two MossFormer2 48-kHz hops.
POLICY = '8s maximum context; 6s stride; retain center, first and last edges; right-zero-pad to hop256; min1024; no adaptive routing'


def layout(frames):
    if not isinstance(frames, int) or frames <= 0:
        raise ValueError('Positive integer sample count required')
    if frames <= WINDOW:
        return max(1024, ((frames + HOP - 1) // HOP) * HOP), [(0, 0, frames)]
    # Ensure the final retained interval reaches the last original sample.
    extra = (frames + MARGIN - WINDOW + STRIDE - 1) // STRIDE
    padded = WINDOW + extra * STRIDE
    parts = []
    cursor = 0
    for start in range(0, padded - WINDOW + 1, STRIDE):
        lo = 0 if start == 0 else MARGIN
        hi = min(WINDOW - MARGIN, frames - start)
        if hi <= lo:
            continue
        if start + lo != cursor:
            raise ValueError('Overlapping or missing retained samples')
        parts.append((start, lo, hi))
        cursor = start + hi
    if cursor != frames:
        raise ValueError('Incomplete last interval')
    return padded, parts


def decode(waveform, infer):
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty mono waveform')
    n = len(x)
    padded, parts = layout(n)
    x = np.pad(x, (0, padded - n))
    result = np.empty(n, dtype=np.float32)
    for start, lo, hi in parts:
        block = x if n <= WINDOW else x[start:start + WINDOW]
        y = np.asarray(infer(block), dtype=np.float32)
        if y.shape != block.shape or not np.isfinite(y).all():
            raise ValueError('Model did not return exact finite padded length')
        result[start + lo:start + hi] = y[lo:hi]
    return result


def to_48k(x):
    return resample_poly(x, 3, 1, window=('kaiser', 8.6)).astype(np.float32)


def to_16k(x, frames):
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (frames * 3,) or not np.isfinite(x).all():
        raise ValueError('Expected exact 48-kHz model output length')
    result = resample_poly(x, 1, 3, window=('kaiser', 8.6)).astype(np.float32)
    if result.shape != (frames,):
        raise ValueError('Resampling changed the expected length')
    return result
