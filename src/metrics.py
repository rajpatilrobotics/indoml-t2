"""Public scorer semantics, with explicit rejection of missing/invalid data.

Source: organizer evaluate.py, revision 32cff11da797b966f74ab4ee21f9b7b38fba85de.
No mean removal or epsilon changes are made to the SI-SDR formula.
"""
import re
import unicodedata
import numpy as np

TAG_RE = re.compile(r'</?[^<>]*>|\[[^\[\]]*\]')


def normalize(text):
    if not isinstance(text, str):
        raise TypeError('Transcript must be a string')
    text = TAG_RE.sub(' ', text)
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.lower().split())


def si_sdr(reference, enhanced):
    ref, enh = np.asarray(reference, dtype=np.float64), np.asarray(enhanced, dtype=np.float64)
    if ref.ndim != 1 or enh.shape != ref.shape or not len(ref):
        raise ValueError('SI-SDR requires equal nonempty mono signals')
    if not np.isfinite(ref).all() or not np.isfinite(enh).all():
        raise ValueError('Nonfinite audio')
    ref_energy = np.dot(ref, ref)
    if ref_energy == 0:
        raise ValueError('SI-SDR undefined for silent reference')
    target = (np.dot(enh, ref) / ref_energy) * ref
    residual = enh - target
    numerator, denominator = np.dot(target, target), np.dot(residual, residual)
    if numerator == denominator == 0:
        raise ValueError('SI-SDR undefined for all-zero estimate')
    with np.errstate(divide='ignore'):
        value = 10 * np.log10(numerator / denominator) if denominator else np.inf
    return float(np.clip(value, -100, 100))


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (word != other)))
        previous = current
    return previous[-1]


def pooled_wer(references, hypotheses, ids):
    edits = words = used = skipped = 0
    for cid in ids:
        if cid not in references or cid not in hypotheses:
            raise ValueError(f'Missing reference or ASR output: {cid}')
        ref = normalize(references[cid]).split()
        if not ref:
            skipped += 1
            continue
        hyp = (normalize(hypotheses[cid]) or '@').split()
        words += len(ref)
        edits += edit_distance(ref, hyp)
        used += 1
    return {'edits': edits, 'reference_words': words, 'clips': used,
            'skipped_empty_references': skipped, 'wer': edits / words if words else None}


def scores(si_mean, references, noisy, enhanced, subsets):
    result = {}
    for name, ids in subsets.items():
        a, b = pooled_wer(references, noisy, ids), pooled_wer(references, enhanced, ids)
        delta = a['wer'] - b['wer'] if a['wer'] is not None and b['wer'] is not None else None
        result[name] = {'noisy': a, 'enhanced': b, 'delta_wer_pp': None if delta is None else 100 * delta}
    delta = result['full']['delta_wer_pp']
    result['mean_synthetic_si_sdr_db'] = si_mean
    result['combined_validation'] = None if si_mean is None or delta is None else si_mean + delta
    syn, nat = result['synthetic']['delta_wer_pp'], result['natural']['delta_wer_pp']
    result['test_weighted_diagnostic'] = None if None in (si_mean, syn, nat) else si_mean + .320345 * syn + .679655 * nat
    return result
