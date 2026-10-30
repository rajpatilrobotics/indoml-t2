"""Deterministic, label-free effective-mask polar projection.

The waveform is reconstructed with overlap normalization and its original length.
No ASR, transcript, clean reference or clip identifier enters this transformation.
"""
import numpy as np

FFT = 512
HOP = 128
RECIPES = {
    'GAPMag90': {'magnitude_exponent': .9, 'phase_strength': 1.},
    'GAPPhase90': {'magnitude_exponent': 1., 'phase_strength': .9},
    'GAPPolar90': {'magnitude_exponent': .9, 'phase_strength': .9},
}


def stft(audio):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Nonempty finite mono waveform required')
    # Explicit zeros cover both edges, including clips shorter than one frame.
    tail = (-len(x)) % HOP
    padded = np.pad(x, (FFT // 2, FFT // 2 + tail))
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::HOP]
    window = np.hanning(FFT + 1)[:-1]
    return np.fft.rfft(frames * window, axis=-1)


def istft(spec, length):
    z = np.asarray(spec)
    expected = 1 + (length + (-length) % HOP) // HOP
    if length <= 0 or z.shape != (expected, FFT // 2 + 1) or not np.isfinite(z).all():
        raise ValueError('Invalid spectrum or requested length')
    window = np.hanning(FFT + 1)[:-1]
    frames = np.fft.irfft(z, n=FFT, axis=-1) * window
    size = (len(frames) - 1) * HOP + FFT
    result, weight = np.zeros(size), np.zeros(size)
    for i, frame in enumerate(frames):
        start = i * HOP
        result[start:start + FFT] += frame
        weight[start:start + FFT] += window ** 2
    section = slice(FFT // 2, FFT // 2 + length)
    if np.any(weight[section] <= 0):
        raise ValueError('Uncovered output samples')
    return result[section] / weight[section]


def project(original, enhanced, magnitude_exponent, phase_strength):
    x, y = np.asarray(original, dtype=np.float64), np.asarray(enhanced, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError('Original and enhanced lengths must match')
    if not (0 < magnitude_exponent <= 1 and 0 < phase_strength <= 1):
        raise ValueError('Projection strengths must be in (0, 1]')
    X, Y = stft(x), stft(y)
    ax, ay = np.abs(X), np.abs(Y)
    # Geometric magnitude interpolation avoids dividing by a near-zero noisy bin.
    # Exact zero magnitudes stay zero; there is no synthetic noise floor.
    mag = np.power(ay, magnitude_exponent) * np.power(ax, 1 - magnitude_exponent)
    # Principal phase difference follows the effective complex mask Y/X.
    delta = np.angle(Y * X.conj())
    angle = np.angle(Y) - (1 - phase_strength) * delta
    result = istft(mag * np.exp(1j * angle), len(x))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite projected audio')
    return result


def transform(original, enhanced, candidate):
    if candidate not in RECIPES:
        raise ValueError('Unknown fixed projection')
    return project(original, enhanced, **RECIPES[candidate])
