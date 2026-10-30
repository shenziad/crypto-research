"""Classical descending-uniform parity sampling, with an explicit finite law.

This is supporting arithmetic for the bounded-output sampler, not a new
Bernoulli primitive. Strict comparisons and fixed truncation are intentional.
"""


def factory(threshold, uniforms, scale):
    assert 0 <= threshold <= scale
    active = True
    parity = False
    previous = threshold
    for uniform in uniforms:
        assert 0 <= uniform < scale
        active = active and uniform < previous
        parity ^= active
        previous = uniform
    return int(not active and not parity), int(active)


def threshold_from_delta(delta_integer, precision=44, output_fraction=20):
    assert precision >= output_fraction+1
    assert -(1 << output_fraction) <= delta_integer <= (1 << output_fraction)
    return ((1 << output_fraction)-delta_integer) << (precision-output_fraction-1)


def acceptance(delta_integer, uniforms_pair, precision=44, output_fraction=20):
    threshold = threshold_from_delta(delta_integer, precision, output_fraction)
    rows = [factory(threshold, coins, 1 << precision) for coins in uniforms_pair]
    return rows[0][0] & rows[1][0], rows[0][1] | rows[1][1]


def first_success(indices, accepted, failure):
    assert len(indices) == len(accepted)
    result = failure
    found = False
    for index, bit in zip(indices, accepted):
        assert bit in (0, 1)
        if bit and not found:
            result = index
        found = found or bit
    return result
