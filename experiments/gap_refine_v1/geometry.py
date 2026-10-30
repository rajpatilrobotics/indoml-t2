"""Apply the same fixed waveform recipe to every clip."""
import numpy as np
from experiments.gap_context_v1.geometry import decode

RECIPES = {
    'GAPShiftMean': {'left_pad_samples': 128, 'baseline_weight': .5,
                     'extra_input': 'original noisy waveform'},
    'GAPRepeat50': {'left_pad_samples': 0, 'baseline_weight': .5,
                    'extra_input': 'serialized Context20 PCM16 waveform'},
}


def refine(original, baseline_pcm, infer, candidate):
    if candidate not in RECIPES:
        raise ValueError('Unknown fixed refinement recipe')
    x = np.asarray(original, dtype=np.float32)
    base = np.asarray(baseline_pcm, dtype=np.float32)
    if (x.ndim != 1 or not x.size or base.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(base).all()):
        raise ValueError('Finite, same-length mono waveforms required')
    recipe = RECIPES[candidate]
    shift = recipe['left_pad_samples']
    if candidate == 'GAPShiftMean':
        shifted = np.pad(x, (shift, 0))
        enhanced = decode(shifted, infer, 'GAPContext20')[shift:shift + len(x)]
    else:
        enhanced = decode(base, infer, 'GAPContext20')
    result = recipe['baseline_weight'] * base + (1 - recipe['baseline_weight']) * enhanced
    if result.shape != x.shape or not np.isfinite(result).all():
        raise ValueError('Invalid refined waveform')
    return result.astype(np.float32, copy=False)
