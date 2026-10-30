"""Train a first bounded residual output adapter; quality pilot, no cryptography.

Frozen public backbone and head; a private candidate adapter would add
b*tanh(A @ B @ normalized_hidden) to its logits. Only final-head adaptation is
studied. It does not implement general internal-layer private LoRA inference.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prereg_path = ROOT/'notes/20261030_bounded_output_adapter_preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    assert sha(Path(__file__).resolve()) == prereg['source_sha256']
    destination = ROOT/'results/bounded_output_adapter_pilot.json'
    assert not destination.exists()
    torch.set_num_threads(2)
    torch.manual_seed(prereg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT/'models/Pythia-14M'
    manifest = json.loads((folder/'download_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(folder/item['file']) == item['sha256']
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to('cuda')
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1 << 30
    datasets = {}
    for name in ('train', 'valid'):
        path = ROOT/'data'/f'ptb_{name}.txt'
        assert sha(path) == prereg['data_sha256'][name]
        datasets[name] = tokenizer(path.read_text(), add_special_tokens=False,
            return_tensors='pt')['input_ids'][0]
    width, vocab, rank = model.config.hidden_size, model.config.vocab_size, prereg['rank']
    def teacher(split, offset):
        ids = datasets[split][offset:offset+129][None].to('cuda')
        assert ids.shape == (1, 129)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True)
            h, z = out.hidden_states[-1][0, :-1].detach().clone(), out.logits[0, :-1].detach().clone()
        return h, z, ids[0, 1:]
    # One training-only window sets a common, fixed hidden-coordinate scale.
    h, _, _ = teacher('train', 0)
    scale = float(h.square().mean().sqrt())
    assert scale > 0
    initial_b = torch.randn(rank, width, device='cuda') / math.sqrt(width)
    adapters, optimizers = {}, {}
    for name in ('linear', 'bounded'):
        a = torch.nn.Parameter(torch.zeros(vocab, rank, device='cuda'))
        b = torch.nn.Parameter(initial_b.clone())
        adapters[name] = (a, b)
        optimizers[name] = torch.optim.Adam([a, b], lr=prereg['learning_rate'])
    def correction(name, h):
        a, b = adapters[name]
        raw = ((h/scale)@b.T)@a.T
        return prereg['bound']*torch.tanh(raw) if name == 'bounded' else raw
    training = []
    started = time.monotonic()
    for step, offset in enumerate(prereg['training_offsets']):
        h, z, target = teacher('train', offset)
        row = dict(step=step, offset=offset, base_nll=float(torch.nn.functional.cross_entropy(z, target)))
        for name in ('linear', 'bounded'):
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            delta = correction(name, h)
            loss = torch.nn.functional.cross_entropy(z+delta, target)
            assert bool(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters[name], max_norm=1.)
            optimizer.step()
            row[name+'_nll_before_update'] = float(loss.detach())
        training.append(row)
        if (step+1) % 8 == 0:
            print('trained', step+1, row, flush=True)
    validation = []
    with torch.no_grad():
        for offset in prereg['validation_offsets']:
            h, z, target = teacher('valid', offset)
            base = z.double()
            row = dict(offset=offset, base_nll=float(torch.nn.functional.cross_entropy(base, target)))
            for name in ('linear', 'bounded'):
                delta = correction(name, h).double()
                row[name+'_nll'] = float(torch.nn.functional.cross_entropy(base+delta, target))
                row[name+'_max_abs_correction'] = float(delta.abs().max())
                if name == 'bounded':
                    assert float(delta.abs().max()) <= prereg['bound']
                    accept = torch.exp(torch.logsumexp(base+delta, -1)-torch.logsumexp(base, -1)-prereg['bound'])
                    assert float(accept.min()) >= math.exp(-2*prereg['bound'])-1e-12
                    assert float(accept.max()) <= 1+1e-12
                    row['acceptance_mean'] = float(accept.mean())
                    row['acceptance_min'] = float(accept.min())
                    row['bounded_TV_vs_base'] = float(.5*(base.softmax(-1)-(base+delta).softmax(-1)).abs().sum(-1).mean())
            validation.append(row)
            print('validation', row, flush=True)
    model_path = ROOT/'results/bounded_output_adapter_weights.npz'
    assert not model_path.exists()
    np.savez_compressed(model_path, **{name+'_'+part: value.detach().cpu().numpy()
        for name, pair in adapters.items() for part, value in zip(('A','B'), pair)}, hidden_scale=scale)
    summary = {name: float(np.mean([row[name+'_nll'] for row in validation])) for name in ('base', 'linear', 'bounded')}
    summary['bounded_improvement'] = summary['base']-summary['bounded']
    summary['linear_improvement'] = summary['base']-summary['linear']
    passed = summary['bounded_improvement'] >= .02 and summary['linear_improvement'] > 0 and summary['bounded_improvement'] >= .5*summary['linear_improvement']
    result = dict(status='completed', model='EleutherAI/pythia-14m', revision=manifest['revision'],
        trained_positions=len(training)*128, validation_positions=len(validation)*128,
        rank=rank, bound=prereg['bound'], hidden_scale=scale, summary=summary,
        pilot_quality_gate_passed=passed, training=training, validation=validation,
        worst_case_single_trial_acceptance_lower_bound=math.exp(-2*prereg['bound']),
        worst_case_trials_for_failure_2neg40=math.ceil(math.log(2**-40)/math.log1p(-math.exp(-2*prereg['bound']))),
        elapsed_seconds_diagnostic=time.monotonic()-started, device=torch.cuda.get_device_name(),
        source_sha256=sha(Path(__file__).resolve()), preregistration_sha256=sha(prereg_path),
        adapter_weights_sha256=sha(model_path), model_sha256=sha(folder/'model.safetensors'),
        limits=prereg['limits'])
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), 'pilot_gate', passed, flush=True)


if __name__ == '__main__':
    main()
