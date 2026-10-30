"""Frozen-profile conformance on reused holdout windows, not fresh quality evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_bounded_fixed16 import OUTPUT_SCALE, encode_parameters, evaluate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registration = ROOT/'notes/20261030_qwen_fixed16_quality_preregistration.json'
    spec = json.loads(registration.read_text())
    destination = ROOT/'results/qwen_fixed16_quality.json'
    assert not destination.exists()
    for name, digest in spec['input_sha256'].items():
        assert sha(ROOT/name) == digest, name
    cert = json.loads((ROOT/'results/qwen_bounded_head_range_certificate.json').read_text())
    assert cert['all_passed']
    eta = cert['error_bounds']['total_delta_uniform']['value']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = ROOT/'models/Qwen2.5-0.5B'
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1 << 30
    with np.load(ROOT/'results/bounded_adapter_qwen_transfer_weights.npz') as archive:
        a = archive['bounded_A'].copy()
        b = archive['bounded_B'].copy()
        scale = float(archive['hidden_scale'])
    ai, bi = encode_parameters(a, b)
    a32 = torch.tensor(a, device='cuda'); b32 = torch.tensor(b, device='cuda')
    a64 = a32.double(); b64 = b32.double()
    previous=json.loads((ROOT/'results/bounded_adapter_qwen_frozen_holdout.json').read_bytes())
    prior={(r['corpus'],r['offset']):r for r in previous['records']}
    records = []
    for corpus, cell in spec['corpora'].items():
        file = ROOT/'data'/cell['file']
        assert sha(file) == cell['sha256']
        tokens = tokenizer(file.read_text(), add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        for offset in cell['offsets']:
            ids = tokens[offset:offset+cell['length']][None].to('cuda')
            assert ids.shape[1] == cell['length']
            with torch.inference_mode():
                output = model(ids, use_cache=False, output_hidden_states=True)
                x32 = output.hidden_states[-1][0, :-1]/scale
                x64 = x32.double()
                logits = output.logits[0, :-1].double(); target = ids[0, 1:]
                delta32 = torch.tanh((x32@b32.T)@a32.T).double()
                delta64 = torch.tanh((x64@b64.T)@a64.T)
                integers, ranges = evaluate(x64.cpu().numpy(), ai, bi)
                finite = torch.tensor(integers, device='cuda', dtype=torch.float64)/OUTPUT_SCALE
                error = float((finite-delta64).abs().max())
                assert error <= eta
                for name, value in ranges.items():
                    assert value <= cert['integer_range_bounds'][name], (name, value)
                ce = torch.nn.functional.cross_entropy
                base_nll = float(ce(logits, target))
                real_nll = float(ce(logits+delta32, target))
                assert abs(base_nll-prior[corpus,offset]['base_nll'])<1e-10
                assert abs(real_nll-prior[corpus,offset]['bounded_nll'])<1e-10
                double_nll = float(ce(logits+delta64, target))
                finite_nll = float(ce(logits+finite, target))
                assert abs(finite_nll-double_nll) <= 2*eta
                p = (logits+delta64).softmax(-1)
                p_finite = (logits+finite).softmax(-1)
                tv = .5*(p-p_finite).abs().sum(-1)
                row = dict(corpus=corpus, offset=offset, positions=len(target),
                    base_nll=base_nll, fp32_adapter_nll=real_nll,
                    float64_adapter_nll=double_nll, finite_adapter_nll=finite_nll,
                    max_abs_delta_error_vs_float64=error,
                    max_abs_fp32_delta_error_vs_float64=float((delta32-delta64).abs().max()),
                    max_input_scaled_l2=float(x64.norm(dim=-1).max()),
                    mean_distribution_tv=float(tv.mean()), max_distribution_tv=float(tv.max()),
                    max_abs_finite_delta=float(finite.abs().max()), integer_range_observed=ranges)
                records.append(row)
                print(json.dumps(row), flush=True)
    summary = {}
    for corpus in spec['corpora']:
        rows = [r for r in records if r['corpus'] == corpus]
        n = sum(r['positions'] for r in rows)
        def mean(name):
            return sum(r['positions']*r[name] for r in rows)/n
        real_gain = mean('base_nll')-mean('fp32_adapter_nll')
        finite_gain = mean('base_nll')-mean('finite_adapter_nll')
        retained = finite_gain/real_gain
        summary[corpus] = dict(positions=n, fp32_gain_nll=real_gain,
            finite_gain_nll=finite_gain, gain_retained=retained,
            quality_gate_passed=bool(real_gain > 0 and finite_gain > 0 and retained >= spec['minimum_gain_retained']),
            max_abs_delta_error_vs_float64=max(r['max_abs_delta_error_vs_float64'] for r in rows),
            max_input_scaled_l2=max(r['max_input_scaled_l2'] for r in rows),
            mean_distribution_tv=mean('mean_distribution_tv'),
            max_distribution_tv=max(r['max_distribution_tv'] for r in rows))
    out = dict(status='completed', conformance_checks_passed=True,
        all_quality_gates_passed=all(s['quality_gate_passed'] for s in summary.values()),
        summary=summary, records=records, preregistration_sha256=sha(registration),
        input_sha256=spec['input_sha256'], limits=spec['limits'])
    destination.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
