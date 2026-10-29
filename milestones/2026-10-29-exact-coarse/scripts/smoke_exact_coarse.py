"""Independent raw-path and exact-function smoke without external assets."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from exact_coarse_spline_check import toy_spec, direct
from exact_coarse_spline import prepare, masked_classifier, evaluate, raw

spec, starts, slopes, intercepts = toy_spec()
a = np.repeat(np.array([0, 1, 127, 4095], dtype=np.uint64), 4)
gamma = np.tile(np.array([0, 1, 7, 15], dtype=np.uint64), 4)
count = 0
for variant in ('basis', 'weighted'):
    keys, _, _ = prepare(len(a), spec, 12, variant, a, gamma)
    for x in range(-32, 32):
        h = (a + np.uint64(x % 4096)) & np.uint64(4095)
        t = sum(masked_classifier(keys[p], h, spec, p) for p in (0, 1)) & np.uint64(15)
        assert np.array_equal(t, (gamma + np.uint64((x // 4) % 16)) & np.uint64(15))
        for party in (0, 1):
            assert np.array_equal(raw(keys[party]['dcf'], h, t, spec, party, 12, variant),
                                  direct(keys[party]['dcf'], h, t, spec, party, 12, variant))
        total = sum(evaluate(keys[p], h, t, spec, p, 12, variant) for p in (0, 1)) & np.uint64(4095)
        index = int(np.searchsorted(starts[1:], x, side='right'))
        expected = np.uint64((slopes[index] * x + intercepts[index]) % 4096)
        if variant == 'weighted':
            rb = (keys[0]['b_mask'] + keys[1]['b_mask']) & np.uint64(4095)
            assert np.all(total[:, 0] == expected)
            assert np.array_equal(total[:, 1], (-rb * expected) & np.uint64(4095))
        else:
            assert np.all(total == expected)
        count += len(a)
print('PASS:', count, 'small-domain input/mask pairs, both layouts, individual-party raw reference checked')
