"""Independent exact enumeration of finite first-accept sampling laws."""
from fractions import Fraction as F
from itertools import product
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p, q = [F(1, 2), F(1, 4), F(1, 4)], [F(1, 4), F(1, 4), F(1, 2)]
    # Four equiprobable proposal words implement q exactly; eight independent
    # coin words implement all acceptance thresholds in these three cases.
    proposals = [0, 1, 2, 2]
    rows = []
    for bound in (1, 2, 4):
        thresholds = [min(F(1), a / (bound*b)) for a, b in zip(p, q)]
        assert all(8*x == int(8*x) for x in thresholds)
        states = [(i, F(coin, 8) < thresholds[i]) for i in proposals for coin in range(8)]
        counts = [0, 0, 0, 0]
        for trials in product(states, repeat=3):
            accepted = next((i for i, accept in trials if accept), 3)
            counts[accepted] += 1
        laws = [F(c, 32**3) for c in counts]
        mass = sum(min(a, bound*b) for a, b in zip(p, q))
        failure = (1-mass/bound)**3
        target = [min(a, bound*b)/mass for a, b in zip(p, q)]
        assert laws[3] == failure
        assert laws[:3] == [(1-failure)*a for a in target]
        tv = sum(abs(a-b) for a, b in zip(p, target))/2
        assert (tv == 0) == (bound >= 2)
        rows.append(dict(bound=bound, exhaustive_trials=32**3,
            exact_output_law=[str(x) for x in laws],
            exact_conditional_target=[str(x) for x in target],
            exact_clipping_tv=str(tv), all_equal=True))
    destination = ROOT/'results/fixed_proposal_sampling_algebra_check.json'
    assert not destination.exists()
    result = dict(all_passed=True, exhaustive_triples=3*32**3, results=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits='Independent finite rational enumeration checks probability algebra only; not a crypto implementation or security proof.')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
