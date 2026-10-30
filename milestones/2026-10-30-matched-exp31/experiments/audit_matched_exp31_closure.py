"""Post-execution replay of frozen E31 experiments; no native processes run.

This is a retrospective audit, not a new preregistered experiment or security
proof. Reconstructing both parties' files is permitted only after termination.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from bounded_adapter_fixed16 import tanh_integer, SCALE
from finite_alias_table import build
from run_dense_adapter_linearot import reconstruct, fixtures as dense_fixtures
from run_dense_adapter_cdf31 import exp31
from run_private_weighted_cdf93_check import audit as cdf_audit
from run_selected_adapter_exp31 import audit_head, fixtures as head_fixtures
from sci_exp31_finite_certificate import exp_enclosure
import run_bounded_sampling_exp31 as rejection

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_bytes())


def main():
    # A specific live native worker would prevent this retrospective audit.
    # Match executable argv, not the audit program's own name or shell text.
    active = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv and Path(argv[0].decode(errors='replace')).name in {
            'dense_adapter_cdf31', 'selected_adapter_exp31', 'private_row_harness',
            'bounded_alias_batch',
        }:
            active.append(str(path.parent))
    assert not active, active

    names = ['dense_adapter_linearot', 'dense_adapter_compact',
             'dense_adapter_ot_chunked', 'public_tanh_ot_native_check',
             'private_weighted_cdf93_check', 'dense_adapter_cdf31',
             'selected_adapter_exp31', 'bounded_sampling_exp31',
             'matched_exp31_full_benchmark']
    verified = {}

    def check_files(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('input_sha256', 'files', 'audit_file_sha256'):
                    for name, digest in item.items():
                        path = ROOT / name
                        assert sha(path) == digest, name
                        if name in verified:
                            assert verified[name] == digest
                        verified[name] = digest
                else:
                    check_files(item)
        elif isinstance(value, list):
            for item in value:
                check_files(item)

    results = {}
    for name in names:
        result = load('results/' + name + '.json')
        assert result['all_passed'], name
        reg = ROOT / ('notes/20261030_' + name + '_preregistration.json')
        # The exhaustive test uses a shorter historical registration filename.
        if name == 'public_tanh_ot_native_check':
            reg = ROOT / 'notes/20261030_public_tanh_ot_native_preregistration.json'
        if name == 'private_weighted_cdf93_check':
            reg = ROOT / 'notes/20261030_private_weighted_cdf93_preregistration.json'
        assert sha(reg) == result['preregistration_sha256'], str(reg)
        check_files(json.loads(reg.read_bytes()))
        check_files(result)
        results[name] = result

    certificate = load('results/sci_exp31_finite_certificate.json')
    assert certificate['all_passed']
    for name, digest in certificate['source_sha256'].items():
        assert sha(ROOT / 'experiments' / name) == digest
    binary = ROOT / 'toolchain/bounded-sampler/sci_exp31_factor_dump'
    assert sha(binary) == certificate['binary_sha256']
    upstream = ROOT / 'third_party/FABLE/extern/EzPC/SCI/src/Math/math-functions.cpp'
    assert sha(upstream) == certificate['upstream_source_sha256']
    actual = json.loads(subprocess.run([str(binary)], capture_output=True,
                                      text=True, check=True).stdout)
    assert actual == certificate['actual_public_factors']
    factors = actual['tables']
    factor_count = 0
    for digit, table in enumerate(factors):
        for j, observed in enumerate(table):
            lower, upper = exp_enclosure(Fraction(j, 1 << (20 - 8 * digit)))
            lo, hi = Fraction(1 << 31) / upper, Fraction(1 << 31) / lower
            assert lo.numerator // lo.denominator == hi.numerator // hi.denominator == observed
            factor_count += 1
    all_delta = np.arange(-(1 << 20), (1 << 20) + 1, dtype=np.int64)
    all_exp = exp31(all_delta, factors)
    assert np.all(np.diff(all_exp) >= 0) and np.all(all_exp > 0)
    assert int(all_exp.max()) == 1 << 31

    def reference(x, b, a):
        rt = x @ b.T
        t = rt // SCALE
        rz = a @ t
        delta = tanh_integer(rz // SCALE)
        return rt, t, rz, delta, exp31(delta, factors)

    def dense_replay(folder, x, b, a, mass):
        paths = [folder / f'party{i}' for i in (1, 2)]
        n = len(a)
        expected = reference(x, b, a)
        sizes = [('raw_t_add41.bin', 41, 8), ('t_add26.bin', 26, 8),
                 ('raw_z_add44.bin', 44, n), ('delta_add44.bin', 44, n),
                 ('exp_add33.bin', 33, n)]
        for (name, width, count), value in zip(sizes, expected):
            for p in paths:
                assert (p / name).stat().st_size == 8 * count
            assert np.array_equal(reconstruct(paths, name, width), value), name
        for p in paths:
            for name, size in [('prefix_add93.bin', 16*n),
                               ('weight_low_add63.bin', 8*n),
                               ('weight_high_add64.bin', 8*n),
                               ('cdf_negative_xor.bin', n),
                               ('r_add93.bin', 16), ('uniform_share.bin', 8)]:
                assert (p / name).stat().st_size == size, (str(p), name)
        replay = cdf_audit(folder, list(map(int, mass)), expected[-1].tolist())
        metrics = [json.loads((p/'metrics.json').read_bytes()) for p in paths]
        for channel in range(3):
            assert metrics[0]['end_counters'][channel] == metrics[1]['end_counters'][channel+3]
            assert metrics[1]['end_counters'][channel] == metrics[0]['end_counters'][channel+3]
        return replay

    with np.load(ROOT/'results/bounded_adapter_gc_inputs.npz') as cap:
        qs = cap['q'].copy()
    dense_replays = 0
    score_values = 0
    for name, x, b, a in dense_fixtures():
        n = len(a)
        if name.startswith(('ptb', 'wiki')):
            q = qs[0 if name.startswith('ptb') else 1]
            mass = build(q, 44)[2][:50304]
        else:
            mass = np.full(n, (1 << 60)//n, dtype=np.uint64)
            mass[:(1 << 60) % n] += 1
        dense_replay(ROOT/'results/dense_adapter_cdf31_runs'/name, x, b, a, mass)
        dense_replays += 1
        # Replay every score-only baseline, not just its all_passed field.
        for variant in ('linearot', 'compact', 'ot_chunked'):
            folder = ROOT / f'results/dense_adapter_{variant}_runs' / name
            expected = reference(x, b, a)
            paths = [folder/f'party{i}' for i in (1, 2)]
            for file, width, value in zip(
                    ('raw_t_add41.bin', 't_add26.bin', 'raw_z_add44.bin', 'delta_add44.bin'),
                    (41, 26, 44, 44), expected):
                assert np.array_equal(reconstruct(paths, file, width), value)
            score_values += n

    head_replays = 0
    for name, x, b, a, indices, mode in head_fixtures():
        folder = ROOT/'results/selected_adapter_exp31_runs'/name
        paths = [folder/f'party{i}' for i in (1, 2)]
        data = [(p/'own_input.bin').read_bytes() for p in paths]
        count = len(a)
        assert all(len(d) == 4608 + 340*count for d in data)
        uniform = [(int.from_bytes(data[0][4608+340*i+36:4608+340*i+44], 'little') ^
                    int.from_bytes(data[1][4608+340*i+36:4608+340*i+44], 'little')) & ((1<<31)-1)
                   for i in range(count)]
        audit_head(paths, x, b, a, indices, uniform)
        head_replays += 1

    rejection_replays = 0
    for corpus in ('ptb', 'wiki'):
        folder = ROOT/'results/bounded_sampling_exp31_runs'/corpus
        with np.load(folder/'client_initial.npz') as archive:
            x, q = archive['x_integer'], archive['q']
        with np.load(folder/'server_initial.npz') as archive:
            a, b = archive['a_integer'], archive['b_integer']
        rejection.audit(folder, x, b, a)
        with np.load(folder/'party0/client_alias_audit.npz') as alias:
            cut, alternate, mass = build(q, 44)[:3]
            assert np.array_equal(alias['cut'], cut)
            assert np.array_equal(alias['alias'], alternate)
            assert np.array_equal(alias['mass'], mass)
        rejection_replays += 1

    bench = results['matched_exp31_full_benchmark']
    common = ROOT/'results/matched_exp31_full_benchmark_runs/common'
    with np.load(common/'client.npz') as archive:
        x, q = archive['x_integer'], archive['q']
    with np.load(common/'server.npz') as archive:
        a, b = archive['a_integer'], archive['b_integer']
    mass = build(q, 44)[2][:50304].astype(np.uint64)
    e = reference(x, b, a)[-1]
    assert sha(common/'client.npz') == sha(ROOT/'results/bounded_sampling_exp31_runs/ptb/client_initial.npz')
    assert sha(common/'server.npz') == sha(ROOT/'results/bounded_sampling_exp31_runs/ptb/server_initial.npz')
    assert hashlib.sha256(mass.astype('<u8').tobytes()).hexdigest() == bench['target_q_mass_sha256']
    assert hashlib.sha256(e.astype('<u8').tobytes()).hexdigest() == bench['target_E31_vector_sha256']
    for row in bench['results']:
        folder = common.parent/f"{row['repetition']}_{row['variant']}"
        if row['variant'] == 'dense':
            dense_replay(folder, x, b, a, mass)
            dense_replays += 1
        else:
            fresh = rejection.audit(folder, x, b, a)
            assert fresh['total_wire_bytes'] == row['total_wire_bytes']
            with np.load(folder/'party0/client_alias_audit.npz') as alias:
                assert np.array_equal(alias['mass'][:50304], mass)
            rejection_replays += 1

    summary = {}
    for variant in ('dense', 'rejection'):
        rows = [r for r in bench['results'] if r['variant'] == variant]
        assert len(rows) == 4
        summary[variant] = dict(observations=len(rows),
            median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in rows),
            median_whole_pair_wall_ms=statistics.median(r['whole_pair_wall_ms'] for r in rows),
            median_participant_elapsed_ms=statistics.median(r['participant_elapsed_ms'] for r in rows))
    assert summary == bench['summary']
    reduction = {key: 100*(1-summary['rejection'][key]/summary['dense'][key])
                 for key in ('median_total_wire_bytes', 'median_whole_pair_wall_ms')}
    out = dict(all_passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        retrospective_audit=True, registered_file_hashes_checked=len(verified),
        score_only_native_pairs_replayed=12, score_only_values_replayed=score_values,
        complete_dense_pairs_replayed=dense_replays,
        complete_rejection_pairs_replayed=rejection_replays,
        selected_head_control_pairs_replayed=head_replays,
        exp_factors_recertified=factor_count, exp_score_points_rechecked=len(all_delta),
        summary=summary, observed_reduction_percent=reduction,
        metadata_errata=[dict(file='results/sci_exp31_finite_certificate.json',
            field='profile.cdf_bits', historical_value=92, implemented_value=93,
            reason='Preimplementation metadata; actual sampler uses93 bits for a signed-difference margin. The factor/error certificate is unchanged.'),
            dict(file='experiments/dense_adapter_cdf31.cpp', field='opening comment',
            reason='Copied score-only comment is stale; actual executable includes private E31 weighted CDF and client-only token release.')],
        evidence_sha256={f'results/{n}.json':sha(ROOT/f'results/{n}.json') for n in names},
        verified_files=verified, source_sha256=sha(Path(__file__)),
        limits='Saved-file replay after verified absence of native workers. Not an independent implementation, new quality experiment, statistical performance confidence interval, or security proof. E31 finite target is distinct from original p44 descending profile. All shares and audit coins remain local; no raw tokens or shares for public upload.')
    destination=ROOT/'results/matched_exp31_closure_audit.json'
    assert not destination.exists()
    destination.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('verified_files','evidence_sha256')}, indent=2))


if __name__ == '__main__':
    main()
