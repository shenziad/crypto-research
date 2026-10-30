"""Cost screen for a canonical rowwise LayerNorm rejection envelope.

Real-model algebra, float64 numerical diagnostics; not interval-certified
implementation, no MPC. In particular no claim that all rejection envelopes
must incur the costs of this one conservative construction.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'models/Pythia-14M/model.safetensors'
    manifest = json.loads((path.parent / 'download_manifest.json').read_text())
    expected = next(x['sha256'] for x in manifest['files'] if x['file'] == path.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    def tensor(name):
        item = header[name]
        assert item['dtype'] == 'F16'
        return np.memmap(path, mode='r', dtype='<f2', offset=8+size+item['data_offsets'][0],
                         shape=tuple(item['shape'])).astype(np.float64)
    weight = tensor('embed_out.weight')
    gamma = tensor('gpt_neox.final_layer_norm.weight')
    beta = tensor('gpt_neox.final_layer_norm.bias')
    v, d = weight.shape
    # Fixed model-only common shift; softmax is invariant to subtracting c^T h.
    centered = weight - weight.mean(0)
    a = centered * gamma
    a -= a.mean(1, keepdims=True)
    b = centered @ beta
    bound = math.sqrt(d) * np.linalg.norm(a, axis=1) + b
    log_c = float(logsumexp(bound))
    # u=0 is feasible for LayerNorm's real domain, attained at constant input.
    # For this declared-domain input acceptance is exactly Z0/C in this envelope.
    log_z0 = float(logsumexp(b))
    log_m0 = log_c - log_z0
    # If M is enormous, K >= M log(1/delta) is asymptotically accurate. Report
    # a rigorous simple necessary K from union bound: Pr[success] <= K/M.
    # Thus success >=1-delta requires K >= (1-delta)*M.
    result = dict(model='EleutherAI/pythia-14m', revision=manifest['revision'],
        vocabulary=v, width=d, model_sha256=expected,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        row_bound_min=float(bound.min()), row_bound_median=float(np.median(bound)),
        row_bound_max=float(bound.max()), log_total_envelope=log_c,
        feasible_zero_u_log_normalizer=log_z0, feasible_zero_u_log_expected_trials=log_m0,
        feasible_zero_u_log10_expected_trials=log_m0/math.log(10),
        necessary_log10_K_for_delta_2neg40=(log_m0+math.log1p(-2**-40))/math.log(10),
        limits='Canonical mean-row-centered, rowwise LayerNorm Cauchy envelope only; not a lower bound for all samplers or all envelopes. Real arithmetic domain, float64 diagnostic evaluation without directed rounding. Zero normalized coordinate is a legitimate domain-cost example, not a natural-corpus frequency claim. No crypto, timing, input recovery or attack experiment. A positive empirical proposal cannot replace a certified all-input envelope.')
    destination = ROOT / 'results/layernorm_rejection_envelope.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
