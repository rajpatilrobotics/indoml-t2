"""One conservative recipe; checkpoint choice uses external development audio only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from src.audio import sha256
from src.data_manifest import save_json
from src.runtime import digest, require_gpu, source_digest
from experiments.predictive_v1.vendor.tfgridnet import TFGridNet
from .data import Pairs, PINS

RECIPE = {'seed': 20261009, 'optimizer_steps': 1000, 'accumulation': 4, 'crop_frames': 64000,
          'learning_rate': 1e-5, 'weight_decay': .01, 'betas': [.8, .99], 'gradient_clip': 3.,
          'dev_interval': 250, 'dev_pairs': 80, 'identity_fraction': .2, 'snr_db': [-5, 20],
          'dtype': 'FP32', 'activation_checkpointing': 'per GridNet block, non-reentrant',
          'loss': '30 complex compressed STFT MSE + 70 magnitude compressed STFT MSE + waveform log-ratio; compression=0.3, FFT512/hop256. Not competition SI-SDR.',
          'selection': 'Lowest fixed external-dev loss including initialization; require 0.5% relative decrease to evaluate an adapted checkpoint.'}


def experiment_hash():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py', '.json')})


def hybrid_loss(pred, clean):
    if pred.shape != clean.shape or not torch.isfinite(pred).all():
        raise ValueError('Invalid model output')
    window = torch.hann_window(512, device=pred.device)
    def spectrum(x):
        z = torch.stft(x, 512, 256, 512, window, return_complex=True)
        mag = z.abs().clamp_min(1e-8)
        compressed = mag.pow(.3)
        return z * mag.pow(-.7), compressed
    p, pm = spectrum(pred); c, cm = spectrum(clean)
    spectral = 30 * torch.mean(torch.view_as_real(p - c).square()) * 2 + 70 * torch.mean((pm - cm).square())
    ratio = torch.linalg.vector_norm(pred, dim=-1) / torch.linalg.vector_norm(pred - clean, dim=-1).clamp_min(1e-7)
    waveform = -2 * torch.log10(ratio.clamp_min(1e-7)).mean()
    return spectral + waveform


def checkpoint_blocks(model):
    for block in model.blocks:
        forward = block.forward
        def wrapped(x, fn=forward):
            return checkpoint(fn, x, use_reentrant=False) if torch.is_grad_enabled() else fn(x)
        block.forward = wrapped


def load_initial(device):
    from huggingface_hub import hf_hub_download
    pins = json.loads(Path(__file__).parents[1].joinpath('predictive_v1/pins.json').read_text())['GAPPredictor']
    path = Path(hf_hub_download(pins['repo'], pins['filename'], revision=pins['revision'], token=False))
    if path.stat().st_size != pins['bytes'] or sha256(path) != pins['sha256']:
        raise ValueError('Initialization checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['cfg'] != pins['config']:
        raise ValueError('Architecture differs from accepted GAP')
    model = TFGridNet(**state['cfg']); model.load_state_dict(state['model'], strict=True)
    model.to(device)
    return model, state['cfg'], pins


def save_state(path, model, optimizer, step, binding, best, history):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'binding': binding,
             'best': best, 'history': history, 'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp = path.with_suffix('.partial.pt'); torch.save(state, tmp); tmp.replace(path)
    save_json(path.with_suffix('.receipt.json'), {'sha256': sha256(path), 'step': step, 'binding': binding})


def restore_state(path, model, optimizer, binding):
    path = Path(path); receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if receipt['sha256'] != sha256(path) or receipt['binding'] != binding:
        raise ValueError('Resume checkpoint or experiment binding changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['binding'] != binding:
        raise ValueError('Wrong resume experiment')
    model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def update(model, optimizer, pairs, step, device, accumulation=4):
    model.train(); optimizer.zero_grad(set_to_none=True); values = []
    for j in range(accumulation):
        x, target = pairs.get(step * accumulation + j)
        x = torch.from_numpy(x)[None].to(device); target = torch.from_numpy(target)[None].to(device)
        loss = hybrid_loss(model(x, sr=16000), target)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        (loss / accumulation).backward(); values.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
    optimizer.step()
    return float(np.mean(values)), float(norm)


def evaluate_dev(model, pairs, device, count):
    model.eval(); values = []
    with torch.inference_mode():
        for i in range(count):
            x, y = pairs.get(i)
            value = hybrid_loss(model(torch.from_numpy(x)[None].to(device), sr=16000), torch.from_numpy(y)[None].to(device))
            if not torch.isfinite(value): raise ValueError('Nonfinite development loss')
            values.append(float(value))
    return float(np.mean(values))


def main(root):
    root = Path(root); require_gpu(); torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed']); torch.cuda.manual_seed_all(RECIPE['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train = Pairs(root, 'train'); dev = Pairs(root, 'dev')
    model, cfg, pins = load_initial('cuda:0'); checkpoint_blocks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'], betas=tuple(RECIPE['betas']), weight_decay=RECIPE['weight_decay'])
    binding = {'recipe': RECIPE, 'data_sha256': sha256(root / 'adapt_sources.json'), 'data_pins_sha256': sha256(PINS),
               'initialization': pins, 'source_hash': source_digest(), 'experiment_hash': experiment_hash()}
    folder = root / 'training'; folder.mkdir(exist_ok=True)
    latest = sorted(folder.glob('step_*.pt'))
    if (folder / 'resume.pt').exists():
        resume_step = json.loads((folder / 'resume.receipt.json').read_text())['step']
        if not latest or resume_step >= json.loads(latest[-1].with_suffix('.receipt.json').read_text())['step']:
            latest.append(folder / 'resume.pt')
    started = time.monotonic()
    if latest:
        state = restore_state(latest[-1], model, optimizer, binding)
        step, best, history = state['step'], state['best'], state['history']
        print('Resumed exact optimizer step:', step, flush=True)
    else:
        step = 0; initial = evaluate_dev(model, dev, 'cuda:0', RECIPE['dev_pairs'])
        best = {'step': 0, 'loss': initial, 'initial_loss': initial}; history = [{'step': 0, 'dev_loss': initial}]
        save_state(folder / 'step_0000.pt', model, optimizer, 0, binding, best, history)
        print('Initial external-development loss:', initial, flush=True)
    torch.cuda.reset_peak_memory_stats(); segment_start = time.monotonic(); segment_step = step
    while step < RECIPE['optimizer_steps']:
        loss, grad = update(model, optimizer, train, step, 'cuda:0', RECIPE['accumulation']); step += 1
        if step == segment_step + 1 or step % 25 == 0:
            torch.cuda.synchronize()
            seconds = time.monotonic() - segment_start
            profile = {'optimizer_step': step, 'total_steps': RECIPE['optimizer_steps'], 'training_loss': loss, 'gradient_norm_before_clip': grad,
                       'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'seconds_per_update_this_session': seconds / (step - segment_step),
                       'training_remaining_hours_estimate_only': (RECIPE['optimizer_steps'] - step) * seconds / (step - segment_step) / 3600}
            save_json(folder / 'progress.json', profile); print(json.dumps(profile), flush=True)
        if step % RECIPE['dev_interval'] == 0:
            value = evaluate_dev(model, dev, 'cuda:0', RECIPE['dev_pairs'])
            history.append({'step': step, 'dev_loss': value})
            if value < best['loss']: best = {'step': step, 'loss': value, 'initial_loss': best['initial_loss']}
            save_state(folder / f'step_{step:04d}.pt', model, optimizer, step, binding, best, history)
            save_json(folder / 'history.json', history); print('External development checkpoint:', history[-1], flush=True)
        if step % 25 == 0:
            save_state(folder / 'resume.pt', model, optimizer, step, binding, best, history)
    selected = best['step'] > 0 and best['initial_loss'] - best['loss'] >= .005 * max(abs(best['initial_loss']), 1.)
    result = {'status': 'training_complete', 'best': best, 'selected_for_official_validation': selected,
              'binding': binding, 'elapsed_seconds_this_session': time.monotonic() - started, 'history': history}
    if selected:
        state = torch.load(folder / f"step_{best['step']:04d}.pt", map_location='cpu', weights_only=True)
        selected_path = folder / 'selected_predictor.pt'
        if selected_path.exists():
            previous = torch.load(selected_path, map_location='cpu', weights_only=True)
            if previous['cfg'] != cfg or set(previous['model']) != set(state['model']) or any(not torch.equal(previous['model'][k], state['model'][k]) for k in state['model']):
                raise ValueError('Existing selected checkpoint differs from the completed training run')
        else:
            torch.save({'cfg': cfg, 'model': state['model']}, selected_path)
        result['selected_checkpoint_sha256'] = sha256(selected_path)
        result['selected_checkpoint_bytes'] = selected_path.stat().st_size
    save_json(folder / 'selection.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'binding'}, indent=2), flush=True)


if __name__ == '__main__':
    main(Path.cwd())
