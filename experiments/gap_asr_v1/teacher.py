"""Training-only frozen SraVaani encoder. Official transcript generation is unchanged."""
import hashlib
import json
from pathlib import Path
import torch
from src.audio import sha256
from src.data_manifest import save_json
from src.runtime import digest


def finite(value, label):
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite '+label)
    return value


def features(model, wav):
    """Same pinned frontend operations; abs handles zero-magnitude gradients safely."""
    p = model._p
    lengths = torch.full((wav.shape[0],), wav.shape[1], dtype=torch.long, device=wav.device)
    seq = lengths // p['hop_length']
    x = torch.cat((wav[:, :1], wav[:, 1:] - p['preemph'] * wav[:, :-1]), dim=1)
    z = torch.stft(x, n_fft=p['n_fft'], hop_length=p['hop_length'], win_length=p['win_length'],
                   window=model._window.float(), center=True, pad_mode='constant', return_complex=True)
    x = torch.matmul(model._fb.float(), z.abs().pow(p['mag_power']))
    x = torch.log(x + p['log_zero_guard_value'])
    mask = torch.arange(x.shape[-1], device=x.device)[None] < seq[:, None]
    denom = mask.sum(1)[:, None]
    mean = torch.where(mask[:, None], x, torch.zeros_like(x)).sum(2) / denom
    var = torch.where(mask[:, None], x-mean[:, :, None], torch.zeros_like(x)).square().sum(2)/(denom-1)
    # Preserve forward normalization while avoiding sqrt'(0) on a constant feature.
    stable = var.clamp_min(1e-12).sqrt()
    std = stable + (var.sqrt()-stable).detach() + p['CONSTANT']
    x = (x-mean[:, :, None])/std[:, :, None]
    return finite(x.masked_fill(~mask[:, None], p['pad_value']), 'ASR frontend'), seq


def distance(predicted, target, lengths):
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError('Encoder geometry changed')
    mask = torch.arange(predicted.shape[-1], device=predicted.device)[None] < lengths[:, None]
    a = torch.nn.functional.normalize(predicted.float(), dim=1, eps=1e-6)
    b = torch.nn.functional.normalize(target.detach().float(), dim=1, eps=1e-6)
    loss = (1-(a*b).sum(1)).masked_select(mask).mean()
    return finite(loss, 'encoder loss')


class Teacher:
    def __init__(self, snapshot, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True).to(device).eval()
        self.model._ensure_loaded()
        self.model.requires_grad_(False)
        self.model.encoder.eval()
        self.device = device

    def encode(self, wav):
        f, lengths = features(self.model, wav)
        e, lens = self.model.encoder(f.to(self.model._io_dtype), lengths)
        return finite(e, 'encoder output'), lens

    def loss(self, predicted, clean):
        with torch.no_grad():
            target, target_len = self.encode(clean)
        encoded, lengths = self.encode(predicted)
        if not torch.equal(lengths, target_len):
            raise ValueError('Teacher sequence lengths disagree')
        return distance(encoded, target, lengths)

    def frozen_hash(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def preflight(self, wav, clean):
        before = self.frozen_hash()
        with torch.no_grad():
            expected, expected_len = self.model.extract_features(wav, torch.tensor([wav.shape[-1]], device=wav.device))
            actual, actual_len = features(self.model, wav)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            assert torch.equal(actual_len, expected_len)
            official = self.model(input_features=expected, feature_lengths=expected_len)
            encoded, lens = self.encode(wav)
            torch.testing.assert_close(encoded, official.last_hidden_state, rtol=5e-4, atol=5e-4)
            assert torch.equal(lens, official.encoder_lengths)
        x = wav.detach().clone().requires_grad_(True)
        loss = self.loss(x, clean)
        grad, = torch.autograd.grad(loss, x)
        finite(grad, 'teacher input gradient')
        if not grad.abs().sum().item() > 0:
            raise ValueError('Frozen teacher gives no enhancement gradient')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError('Teacher parameters must stay frozen')
        if before != self.frozen_hash():
            raise ValueError('Teacher changed during preflight')
        return {'status':'PASS','frontend_and_encoder_parity':True,'input_gradient_norm':float(grad.norm()),
                'teacher_weights_sha256':before,'teacher_parameters_updated':False}


def prepare_snapshot(root, token):
    from huggingface_hub import snapshot_download
    spec = json.loads((Path(root)/'configs/phase2_gap_adapt.json').read_text())['validated_asr_config']
    path = Path(snapshot_download(spec['asr_repo'], revision=spec['asr_revision'], token=token,
                                 allow_patterns=['*.ts','*.pt','*.model','*.json','*.py']))
    found={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.suffix in ('.ts','.pt','.model','.json','.py')}
    if found != spec['artifact_sha256']:
        raise ValueError('Teacher artifacts differ from frozen mandated ASR')
    save_json(Path(root)/'teacher_snapshot.json',{'path':str(path),'artifact_sha256':found,'revision':spec['asr_revision']})
    return path


if __name__ == '__main__':
    import sys
    prepare_snapshot(Path.cwd(), sys.stdin.read().strip())
    print('Pinned frozen ASR snapshot verified.', flush=True)
