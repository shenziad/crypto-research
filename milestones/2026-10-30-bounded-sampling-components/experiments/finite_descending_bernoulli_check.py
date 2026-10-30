"""Exact rational law, truncation and fixed-batch sampling budget checks."""
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path

from finite_descending_bernoulli import factory, threshold_from_delta, first_success

ROOT = Path(__file__).resolve().parents[1]


def rep(x):
    return dict(numerator=x.numerator, denominator=x.denominator, value=float(x))


def exp_upper(x):
    term = F(1); lower = term
    for k in range(1, 129):
        term *= x/k
        lower += term
    assert x < 130
    return lower+(term*x/129)/(1-x/130)


def main():
    destination = ROOT/'results/finite_descending_bernoulli_check.json'
    assert not destination.exists()
    enumeration_cases = 0
    for s in (1, 2, 4):
        for m in range(s+1):
            accepted = 0
            for coins in itertools.product(range(s), repeat=s+1):
                out, overflow = factory(m, coins, s)
                assert not overflow
                accepted += out
                enumeration_cases += 1
            assert F(accepted, s**(s+1)) == (1-F(1, s))**m
    dp_cells = 0
    for s in (1, 2, 4, 8, 16, 64, 256):
        even = [F(0)]*(s+1); odd = even.copy(); overflow = [F(1)]*(s+1)
        for length in range(1, s+2):
            en = []; on = []; rn = []
            es = os = rs = F(0)
            for m in range(s+1):
                en.append(F(s-m, s)+os/s)
                on.append(es/s)
                rn.append(rs/s)
                assert en[-1]+on[-1]+rn[-1] == 1
                expected_overflow = F(math.comb(m, length), s**length) if m >= length else F(0)
                assert rn[-1] == expected_overflow <= F(1, math.factorial(length))
                exact = (1-F(1, s))**m
                assert 0 <= exact-en[-1] <= rn[-1]
                if length == s+1:
                    assert en[-1] == exact
                es += even[m]; os += odd[m]; rs += overflow[m]
                dp_cells += 1
            even, odd, overflow = en, on, rn
    # All 2^21+1 finite output grid points map exactly, endpoints included.
    p = 44; s = 1 << p; f = 20; delta_checks = 0
    for delta in range(-(1 << f), (1 << f)+1):
        m = threshold_from_delta(delta, p, f)
        assert F(2*m, s) == 1-F(delta, 1 << f)
        delta_checks += 1
    # First-success identity with an explicit failure symbol; no acceptance
    # flags/proposal indices need to be public for this finite functionality.
    selection_cases = 0
    for length in range(1, 9):
        indices = list(range(length))
        for accepted in itertools.product((0, 1), repeat=length):
            expected = next((i for i, b in enumerate(accepted) if b), 65536)
            assert first_success(indices, accepted, 65536) == expected
            selection_cases += 1
    for q, a in [([F(1, 3), F(2, 3)], [F(1, 5), F(4, 5)]),
                 ([F(0), F(1, 2), F(1, 2)], [F(1), F(1, 4), F(1, 2)])]:
        alpha = sum((x*y for x, y in zip(q, a)), F(0))
        for k in range(1, 13):
            factor = sum(((1-alpha)**j for j in range(k)), F(0))
            mass = [x*y*factor for x, y in zip(q, a)]
            failure = (1-alpha)**k
            assert sum(mass)+failure == 1
            assert all(v/(1-failure) == x*y/alpha for v, x, y in zip(mass, q, a))
    # log(1-x)>=-x/(1-x) gives a_min >= exp(-2S/(S-1)).
    alpha_lower = F(1353, 10000)
    assert 1/exp_upper(F(2*s, s-1)) > alpha_lower
    k = 201; length = 19
    failure_bound = (1-alpha_lower)**k
    truncation_bound = F(2*k, math.factorial(length))
    # c=-S log(1-1/S), c-1<=1/[2(S-1)]. The normalized
    # exponential tilt ratio is within exp(+-2(c-1)); e^x<=1/(1-x).
    temperature_tv_bound = F(1, s-2)
    total = failure_bound+truncation_bound+temperature_tv_bound
    assert total < F(1, 1 << 40)
    names = ['finite_descending_bernoulli.py', 'finite_descending_bernoulli_check.py']
    out = dict(all_passed=True, exhaustive_coin_states=enumeration_cases,
        exact_rational_dp_cells=dp_cells, finite_delta_grid_points_checked=delta_checks,
        first_success_cases=selection_cases,
        finite_law='Single untruncated strict-descent factory accepts with (1-1/S)^m, not exp(-m/S). Two independent factories induce q_i exp(c_S delta_i), c_S=-S log(1-1/S).',
        fixed_profile=dict(precision=p, output_fraction=f, proposals=k, draws_per_factory=length,
                           independent_factories_per_proposal=2, minimum_acceptance=rep(alpha_lower)),
        sampling_error_budget=dict(no_success=rep(failure_bound),
            factory_truncation_coupling=rep(truncation_bound), temperature_tv=rep(temperature_tv_bound),
            total=rep(total), target=rep(F(1, 1 << 40))),
        source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in names},
        prior='Classical von Neumann/Forsythe descending-uniform method; Sun et al. Cybersecurity2022 DOI10.1186/s42400-022-00113-0 Algorithms2/3 already use exponential Bernoulli sampling and analyze finite precision. No independent primitive/newness claim.',
        limits='Exact finite mathematical conformance only. Total bound is relative to the exact finite proposal q and bounded finite delta score, excluding alias-q approximation, score error, hash/cuckoo failures and computational security. Requires independent uniform coins and independent proposals; fixed201 replaces ideal-only191 for this separate finite profile. No actual MPC, model speed or complete protocol security claim.')
    destination.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({key:out[key] for key in ('all_passed', 'exhaustive_coin_states',
        'exact_rational_dp_cells', 'sampling_error_budget')}, indent=2))


if __name__ == '__main__':
    main()
