"""A single length-based context policy per candidate, independent of labels."""
import numpy as np
from experiments.predictive_v1.geometry import layout as parent_layout

POLICIES = {
    'GAPContext4': {'window': 64000, 'stride': 48000, 'margin': 8000},
    'GAPContext20': {'window': 320000, 'stride': 288000, 'margin': 16000},
}
HOP = 256


def layout(frames, candidate):
    if candidate not in POLICIES:
        raise ValueError('Unknown fixed context policy')
    if not isinstance(frames, int) or frames <= 0:
        raise ValueError('Positive integer sample count required')
    policy = POLICIES[candidate]
    window, stride, margin = (policy[k] for k in ('window', 'stride', 'margin'))
    if frames <= window:
        return max(1024, ((frames + HOP - 1) // HOP) * HOP), [(0, 0, frames)]
    extra = (frames + margin - window + stride - 1) // stride
    padded = window + extra * stride
    parts, cursor = [], 0
    for start in range(0, padded - window + 1, stride):
        lo = 0 if start == 0 else margin
        hi = min(window - margin, frames - start)
        if hi <= lo:
            continue
        if start + lo != cursor:
            raise ValueError('Context coverage gap or overlap')
        parts.append((start, lo, hi))
        cursor = start + hi
    if cursor != frames:
        raise ValueError('Incomplete final context')
    return padded, parts


def forward_layout(frames, candidate=None):
    if candidate is None:
        padded, parts = parent_layout(frames)
        window = 128000
    else:
        padded, parts = layout(frames, candidate)
        window = POLICIES[candidate]['window']
    return (padded, [(start, min(window, padded) if frames > window else padded, lo, hi)
                     for start, lo, hi in parts])


def can_reuse_parent(frames, candidate):
    return forward_layout(frames, candidate) == forward_layout(frames)


def decode(waveform, infer, candidate):
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty mono waveform')
    n = len(x)
    padded, parts = layout(n, candidate)
    window = POLICIES[candidate]['window']
    x = np.pad(x, (0, padded - n))
    result = np.empty(n, dtype=np.float32)
    for start, lo, hi in parts:
        block = x if n <= window else x[start:start + window]
        y = np.asarray(infer(block), dtype=np.float32)
        if y.shape != block.shape or not np.isfinite(y).all():
            raise ValueError('Model returned invalid shape or nonfinite audio')
        result[start + lo:start + hi] = y[lo:hi]
    return result
