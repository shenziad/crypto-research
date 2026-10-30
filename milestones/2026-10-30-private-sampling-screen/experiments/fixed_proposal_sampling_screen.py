"""Optimistic numerical rejection-sampling filter, not a secure protocol.

All logits/normalizers are given for free. Secret row selection, exp, comparison,
first-success and protocol traffic are ignored: a positive result is only a
necessary cost screen. Public proposals are NOT asserted to hide output tokens.
"""
import datetime
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / 'notes/20261030_fixed_proposal_sampling_preregistration.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.quantile(a, .95)), max=float(a.max()))


def trials(m, delta):
    if m <= 1:
        return 1
    return math.ceil(math.log(delta) / math.log1p(-1 / m))


def main():
    torch.set_num_threads(1)
    prereg = json.loads(PREREG.read_text())
    assert sha(Path(__file__).resolve()) == prereg['source_sha256']
    destination = ROOT / 'results/fixed_proposal_sampling_screen.json'
    assert not destination.exists()
    start = time.monotonic()
    model_dir = ROOT / 'models/Pythia-14M'
    manifest = json.loads((model_dir / 'download_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(model_dir / item['file']) == item['sha256']
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False)
    tokenizer.model_max_length = 1 << 30
    vocabulary = model.config.vocab_size
    tokens, hashes = {}, {}
    for corpus, filename in [('wiki', 'wikitext2_test.txt'), ('ptb', 'ptb_test.txt')]:
        path = ROOT / 'data' / filename
        hashes[corpus] = sha(path)
        assert hashes[corpus] == prereg['data_sha256'][corpus]
        tokens[corpus] = tokenizer(path.read_text(), add_special_tokens=False,
            return_tensors='pt')['input_ids'][0]

    def probabilities(corpus, offset):
        ids = tokens[corpus][offset:offset + 65]
        assert len(ids) == 65
        with torch.inference_mode():
            logits = model(ids[None], use_cache=False).logits[0, :-1].double()
            p = logits.softmax(-1).numpy()
        assert p.shape == (64, vocabulary) and np.all(p > 0)
        assert np.max(np.abs(p.sum(1) - 1)) < 1e-12
        return p, ids.numpy()

    train_sum = np.zeros(vocabulary)
    train_max = np.zeros(vocabulary)
    inputs = {}
    calibration = []
    for j, offset in enumerate(prereg['calibration_wiki_offsets']):
        p, ids = probabilities('wiki', offset)
        train_sum += p.sum(0)
        train_max = np.maximum(train_max, p.max(0))
        inputs[f'calibration_{j}'] = ids
        calibration.append(p)
    smoothing = prereg['uniform_mixture']
    proposals = {'uniform': np.full(vocabulary, 1 / vocabulary),
        'calibration_mean': (1 - smoothing) * train_sum / train_sum.sum() + smoothing / vocabulary,
        'calibration_minimax': (1 - smoothing) * train_max / train_max.sum() + smoothing / vocabulary}
    envelopes = {name: max(float((p / q).max()) for p in calibration)
                 for name, q in proposals.items()}
    del calibration
    all_max = train_max.copy()
    heldout_max = np.zeros(vocabulary)
    rows = []
    for corpus in ('wiki', 'ptb'):
        for j, offset in enumerate(prereg[f'heldout_{corpus}_offsets']):
            p, ids = probabilities(corpus, offset)
            inputs[f'{corpus}_{j}'] = ids
            heldout_max = np.maximum(heldout_max, p.max(0))
            all_max = np.maximum(all_max, p.max(0))
            for name, q in proposals.items():
                ratio = p / q
                m = ratio.max(1)
                # This calibration-only envelope need not cover a fresh context.
                b = envelopes[name]
                clipped = np.minimum(p, b * q)
                mass = clipped.sum(1)
                qclip = clipped / mass[:, None]
                tv = .5 * np.abs(p - qclip).sum(1)
                k = trials(b, 2 ** -40)
                failure = np.exp(k * np.log1p(-np.minimum(mass / b, 1)))
                rows.append(dict(corpus=corpus, offset=offset, proposal=name,
                    oracle_context_envelopes=m.tolist(),
                    oracle_K_delta_2neg40=[trials(float(x), 2 ** -40) for x in m],
                    calibration_envelope=b, calibration_K_delta_2neg40=k,
                    positions_exceeding_calibration_envelope=int(np.count_nonzero(m > b * (1 + 1e-12))),
                    clipped_target_tv=tv.tolist(), clipped_mass=mass.tolist(),
                    no_accept_probability_with_calibration_K=failure.tolist()))
            print(corpus, offset, '64 distributions evaluated', flush=True)
    aggregates = {}
    for corpus in ('wiki', 'ptb'):
        for name in proposals:
            group = [r for r in rows if r['corpus'] == corpus and r['proposal'] == name]
            m = [v for r in group for v in r['oracle_context_envelopes']]
            ks = [v for r in group for v in r['oracle_K_delta_2neg40']]
            aggregates[f'{corpus}/{name}'] = dict(oracle_envelope=summary(m),
                oracle_K_delta_2neg40=summary(ks),
                fractions_K_at_least_V=float(np.mean(np.asarray(ks) >= vocabulary)),
                calibration_envelope=envelopes[name],
                calibration_K_delta_2neg40=trials(envelopes[name], 2 ** -40),
                fresh_positions=len(m),
                positions_exceeding_calibration_envelope=sum(r['positions_exceeding_calibration_envelope'] for r in group),
                clipped_target_tv=summary([v for r in group for v in r['clipped_target_tv']]),
                clipped_mass=summary([v for r in group for v in r['clipped_mass']]))
    archive = ROOT / 'results/fixed_proposal_sampling_inputs.npz'
    assert not archive.exists()
    np.savez_compressed(archive, **inputs, **{f'proposal_{k}': v for k, v in proposals.items()},
                        calibration_pointwise_max=train_max, all_pointwise_max=all_max,
                        heldout_pointwise_max=heldout_max)
    # For a finite set of targets, min_q max_{t,i} p_t(i)/q(i) = sum_i max_t p_t(i).
    oracle_minimax = {}
    for name, envelope in [('calibration', train_max), ('heldout', heldout_max), ('all', all_max)]:
        m = float(envelope.sum())
        oracle_minimax[name] = dict(M=m, K_delta_2neg40=trials(m, 2 ** -40),
            K_over_V=trials(m, 2 ** -40) / vocabulary,
            meaning='Oracle optimal fixed proposal for this finite observed set; uses heldout data where named. Not a deployable or all-input envelope.')
    result = dict(status='completed', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model='EleutherAI/pythia-14m', revision=manifest['revision'], vocabulary=vocabulary,
        framework='Transformers FP32 CPU eager; softmax evaluated float64',
        calibration_positions=512, heldout_positions=1024, aggregates=aggregates,
        observed_set_oracle_minimax=oracle_minimax, rows=rows,
        elapsed_seconds_diagnostic=time.monotonic() - start,
        source_sha256=sha(Path(__file__).resolve()), preregistration_sha256=sha(PREREG),
        input_archive_sha256=sha(archive), data_sha256=hashes,
        model_sha256=sha(model_dir / 'model.safetensors'),
        limits=prereg['limits'])
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(aggregates=aggregates, oracle_minimax=oracle_minimax), indent=2))


if __name__ == '__main__':
    main()
