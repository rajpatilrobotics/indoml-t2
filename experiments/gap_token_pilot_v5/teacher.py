"""Training-only clean-path posterior matching; no recognizer weights are trained."""
import json
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from experiments.gap_asr_v1.teacher import Teacher as EncoderTeacher, finite

OBJECTIVE = {
    'name': 'clean greedy TDT path posterior KL',
    'temperature': 2.0,
    'duration_weight': 0.25,
    'max_emission_positions': 12,
    'max_blank_positions': 12,
    'description': 'Token and duration distributions normalized separately; detached clean states and logits. '
                   'This sparse clean-path surrogate is not full RNNT/TDT sequence likelihood.',
}


def posterior_kl(student, clean, num_durations, temperature=2.0):
    if student.shape != clean.shape or student.ndim != 1 or not 0 < num_durations < len(student):
        raise ValueError('Invalid TDT joint geometry')
    finite(student, 'student joint'); finite(clean, 'clean joint')
    def divergence(a, b):
        log_a = torch.log_softmax(a.float() / temperature, -1)
        log_b = torch.log_softmax(b.detach().float() / temperature, -1)
        return (log_b.exp() * (log_b - log_a)).sum() * temperature**2
    return divergence(student[:-num_durations], clean[:-num_durations]) + OBJECTIVE['duration_weight'] * divergence(student[-num_durations:], clean[-num_durations:])


def spread_indices(indices, count):
    if len(indices) <= count:
        return list(indices)
    # Deterministic coverage, without choosing positions using validation outcomes.
    return [indices[(j * (len(indices) - 1)) // (count - 1)] for j in range(count)]


def selected_positions(path, blank):
    emitted = [i for i, p in enumerate(path) if p['token'] != blank]
    blanks = [i for i, p in enumerate(path) if p['token'] == blank]
    return sorted(spread_indices(emitted, OBJECTIVE['max_emission_positions']) +
                  spread_indices(blanks, OBJECTIVE['max_blank_positions']))


@torch.no_grad()
def trace_clean(model, encoded, frames):
    """Mirror the pinned official _greedy_one, retaining *pre-call* decoder states."""
    cfg = model.config; dev = encoded.device
    if encoded.shape[0] != 1 or not 0 < frames <= encoded.shape[-1]:
        raise ValueError('Invalid clean encoder sequence')
    h = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=model._io_dtype)
    c = torch.zeros_like(h); last = cfg.blank_id; path = []; tokens = []; times = []
    lengths = torch.ones(1, dtype=torch.int32, device=dev); t = 0
    while t < frames:
        frame = encoded.narrow(2, t, 1); added = 0; need = True
        while need and added < cfg.max_symbols:
            target = torch.tensor([[last]], dtype=torch.int32, device=dev)
            logits, _, h2, c2 = model.decoder_joint(frame, target, lengths, h, c)
            if logits.ndim != 4 or tuple(logits.shape[:3]) != (1, 1, 1):
                raise ValueError('Unexpected decoder joint shape')
            logits = finite(logits[0, 0, 0], 'clean joint')
            token = int(logits[:-cfg.num_durations].argmax())
            skip = cfg.durations[int(logits[-cfg.num_durations:].argmax())]
            path.append({'frame': t, 'last': last, 'h': h.detach().clone(), 'c': c.detach().clone(),
                         'logits': logits.detach().clone(), 'token': token})
            if token != cfg.blank_id:
                tokens.append(token); times.append(t); h, c, last = h2, c2, token
            added += 1; t += skip; need = skip == 0
        if added == cfg.max_symbols:
            t += 1
    return path, tokens, times


class Teacher(EncoderTeacher):
    def encode(self, wav):
        # This scoped backend change applies only to the differentiable training teacher.
        # Both checkpoint forward and recomputation call this same function.
        from .native_encoder import encode_native
        return encode_native(self, wav)

    def path_loss(self, predicted_encoded, path):
        positions = selected_positions(path, self.model.config.blank_id)
        if not positions:
            raise ValueError('Empty clean teacher path')
        losses = []
        lengths = torch.ones(1, dtype=torch.int32, device=predicted_encoded.device)
        for index in positions:
            entry = path[index]
            target = torch.tensor([[entry['last']]], dtype=torch.int32, device=predicted_encoded.device)
            logits, _, _, _ = self.model.decoder_joint(predicted_encoded.narrow(2, entry['frame'], 1),
                                                       target, lengths, entry['h'], entry['c'])
            losses.append(posterior_kl(logits[0, 0, 0], entry['logits'], self.model.config.num_durations,
                                       OBJECTIVE['temperature']))
        return finite(torch.stack(losses).mean(), 'posterior loss')

    def loss(self, predicted, clean):
        with torch.no_grad():
            target, target_len = self.encode(clean)
            path, _, _ = trace_clean(self.model, target, int(target_len[0]))
        # Recompute the frozen encoder during backward to limit GPU memory.
        encoded, lengths = checkpoint(self.encode, predicted, use_reentrant=False)
        if not torch.equal(lengths, target_len):
            raise ValueError('Clean/enhanced encoder alignment mismatch')
        return self.path_loss(encoded, path)

    def preflight(self, wav, clean, diagnostic_folder=None):
        before = self.frozen_hash()
        with torch.no_grad():
            expected, expected_len = self.model.extract_features(clean, torch.tensor([clean.shape[-1]], device=clean.device))
            official = self.model(input_features=expected, feature_lengths=expected_len)
            target, lens = self.encode(clean)
            torch.testing.assert_close(target, official.last_hidden_state, rtol=5e-4, atol=5e-4)
            assert torch.equal(lens, official.encoder_lengths)
            path, tokens, times = trace_clean(self.model, target, int(lens[0]))
            official_tokens, official_times = self.model._greedy_one(target, int(lens[0]))
            if tokens != official_tokens or times != official_times:
                raise ValueError('Training clean path differs from official greedy decoder')
            same = self.path_loss(target, path)
            if abs(float(same)) > 1e-5:
                raise ValueError('Identical clean posterior KL should be zero')
        from .diagnostics import repeated_gradient_check
        gradient = repeated_gradient_check(self, wav, clean, diagnostic_folder or Path('numeric_diagnostics')/'teacher_preflight')
        if float(gradient.norm()) <= 0:
            raise ValueError('No posterior gradient reaches enhanced audio')
        if self.frozen_hash() != before or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError('Frozen recognizer changed')
        return {'status': 'PASS', 'clean_path_matches_official_decoder': True, 'encoder_parity': True,
                'clean_token_count': len(tokens), 'clean_path_positions': len(path),
                'used_positions': len(selected_positions(path, self.model.config.blank_id)),
                'same_input_KL': float(same), 'input_gradient_norm': float(gradient.norm()),
                'teacher_weights_sha256': before, 'teacher_parameters_updated': False, 'objective': OBJECTIVE, 'encoder_backend': getattr(self, 'encoder_backend_evidence', {})}
