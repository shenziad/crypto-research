"""Fixed-hyperparameter quality transfer to Qwen0.5B; no private protocol.

Train/validate once. Evaluate frozen test windows only if the preregistered
validation gate passes. Failure retires this exact transfer, not every adapter.
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg = ROOT/'notes/20261030_bounded_adapter_qwen_transfer_preregistration.json'
    spec = json.loads(reg.read_bytes())
    assert sha(Path(__file__)) == spec['source_sha256']
    destination = ROOT/'results/bounded_adapter_qwen_transfer.json'
    weights_path = ROOT/'results/bounded_adapter_qwen_transfer_weights.npz'
    assert not destination.exists() and not weights_path.exists()
    for name, digest in spec['model_files_sha256'].items():
        assert sha(ROOT/'models/Qwen2.5-0.5B'/name) == digest, name
    torch.set_num_threads(2)
    torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    folder = ROOT/'models/Qwen2.5-0.5B'
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1 << 30
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to('cuda')
    model.requires_grad_(False)
    datasets = {}

    def get_dataset(name):
        if name not in datasets:
            item = spec['data'][name]
            p = ROOT/'data'/item['file']
            assert sha(p) == item['sha256']
            datasets[name] = tokenizer(p.read_text(), add_special_tokens=False,
                return_tensors='pt')['input_ids'][0]
        return datasets[name]

    def teacher(split, offset):
        ids = get_dataset(split)[offset:offset+129][None].to('cuda')
        assert ids.shape == (1, 129), (split, offset, ids.shape)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True)
            return out.hidden_states[-1][0, :-1].detach().clone(), out.logits[0, :-1].detach().clone(), ids[0, 1:]

    started = time.monotonic()
    h, _, _ = teacher('train', 0)
    scale = float(h.square().mean().sqrt())
    assert scale > 0 and math.isfinite(scale)
    width, vocab, rank = model.config.hidden_size, model.config.vocab_size, spec['rank']
    initial_b = torch.randn(rank, width, device='cuda')/math.sqrt(width)
    adapters, optimizers = {}, {}
    for name in ('linear', 'bounded'):
        a = torch.nn.Parameter(torch.zeros(vocab, rank, device='cuda'))
        b = torch.nn.Parameter(initial_b.clone())
        adapters[name] = (a, b)
        optimizers[name] = torch.optim.Adam([a, b], lr=spec['learning_rate'])

    def correction(name, h):
        a, b = adapters[name]
        raw = ((h/scale)@b.T)@a.T
        return spec['bound']*torch.tanh(raw) if name == 'bounded' else raw

    training = []
    for step, offset in enumerate(spec['training_offsets']):
        h, z, target = teacher('train', offset)
        row = dict(step=step, offset=offset,
                   base_nll=float(torch.nn.functional.cross_entropy(z, target)))
        for name in ('linear', 'bounded'):
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(z+correction(name, h), target)
            assert bool(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters[name], max_norm=1.)
            optimizer.step()
            row[name+'_nll_before_update'] = float(loss.detach())
        training.append(row)
        if (step+1) % 8 == 0:
            print('trained', step+1, row, flush=True)
    # Freeze the sole final checkpoint before validation/test evaluation.
    for pair in adapters.values():
        for parameter in pair:
            parameter.requires_grad_(False)
    np.savez_compressed(weights_path,
        **{name+'_'+part: tensor.detach().cpu().numpy()
           for name, pair in adapters.items() for part, tensor in zip(('A', 'B'), pair)},
        hidden_scale=scale)

    def evaluate(split, offsets):
        records = []
        for offset in offsets:
            h, z, target = teacher(split, offset)
            with torch.no_grad():
                base = z.double()
                row = dict(split=split, offset=offset, positions=len(target),
                    base_nll=float(torch.nn.functional.cross_entropy(base, target)),
                    max_normalized_input_l2=float(torch.linalg.vector_norm(h/scale, dim=-1).max()))
                for name in ('linear', 'bounded'):
                    delta = correction(name, h).double()
                    row[name+'_nll'] = float(torch.nn.functional.cross_entropy(base+delta, target))
                    row[name+'_max_abs_correction'] = float(delta.abs().max())
                    if name == 'bounded':
                        assert float(delta.abs().max()) <= spec['bound']
                        acceptance = (torch.logsumexp(base+delta, -1)-torch.logsumexp(base, -1)-spec['bound']).exp()
                        assert float(acceptance.min()) >= math.exp(-2*spec['bound'])-1e-12
                        assert float(acceptance.max()) <= 1+1e-12
                        row['mean_acceptance'] = float(acceptance.mean())
                        row['min_acceptance'] = float(acceptance.min())
                records.append(row)
            print('evaluate', split, offset, 'bounded_minus_base', row['bounded_nll']-row['base_nll'], flush=True)
        return records

    def summarize(rows):
        d = np.array([r['bounded_nll']-r['base_nll'] for r in rows])
        linear = np.array([r['linear_nll']-r['base_nll'] for r in rows])
        rng = np.random.default_rng(spec['seed'])
        boot = d[rng.integers(0, len(rows), (20000, len(rows)))].mean(1)
        return dict(positions=sum(r['positions'] for r in rows),
            base_nll=float(np.mean([r['base_nll'] for r in rows])),
            bounded_minus_base_nll=float(d.mean()), linear_minus_base_nll=float(linear.mean()),
            gain_retained_vs_linear=float(d.mean()/linear.mean()) if linear.mean()<0 else None,
            descriptive_reset_window_bootstrap95=np.quantile(boot, [.025,.975]).tolist(),
            mean_acceptance=float(np.mean([r['mean_acceptance'] for r in rows])),
            min_acceptance=min(r['min_acceptance'] for r in rows),
            max_normalized_input_l2=max(r['max_normalized_input_l2'] for r in rows))

    validation = evaluate('valid', spec['validation_offsets'])
    val = summarize(validation)
    passed = (val['bounded_minus_base_nll'] <= -.02 and
              val['linear_minus_base_nll'] < 0 and val['gain_retained_vs_linear'] >= .5)
    print('fixed_validation_gate', passed, val, flush=True)
    holdout, holdout_summary = [], {}
    if passed:
        for split in ('ptb_test', 'wiki_test'):
            rows = evaluate(split, spec['data'][split]['offsets'])
            holdout.extend(rows)
            holdout_summary[split] = summarize(rows)
    # Both corpora must improve by >=.01 nat and retain >=half the positive
    # linear-control gain. This is a go/no-go filter, not significance testing.
    transfer_passed = passed and all(
        s['bounded_minus_base_nll'] <= -.01 and
        (s['linear_minus_base_nll'] >= 0 or s['gain_retained_vs_linear'] >= .5)
        for s in holdout_summary.values())
    outcome = dict(status='completed', model='Qwen/Qwen2.5-0.5B',
        trained_positions=128*len(training), rank=rank, bound=spec['bound'],
        hidden_scale=scale, validation_gate_passed=passed,
        fixed_transfer_gate_passed=transfer_passed, validation_summary=val,
        holdout_summary=holdout_summary, training=training, validation=validation, holdout=holdout,
        adapter_max_abs={name+'_'+part:float(tensor.abs().max())
            for name, pair in adapters.items() for part, tensor in zip(('A','B'),pair)},
        model_shape=dict(hidden=width, vocab=vocab), device=torch.cuda.get_device_name(),
        elapsed_seconds_diagnostic=time.monotonic()-started,
        source_sha256=sha(Path(__file__)), preregistration_sha256=sha(reg),
        model_sha256=sha(folder/'model.safetensors'), weights_sha256=sha(weights_path),
        decision='Advance only if fixed transfer gate passes; otherwise archive this exact larger-model transfer without rank/bound/LR/seed retries.',
        limits=spec['limits'])
    destination.write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps({k:outcome[k] for k in ('validation_gate_passed','fixed_transfer_gate_passed','validation_summary','holdout_summary')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
