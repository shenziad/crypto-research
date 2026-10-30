"""Compile a floating distribution to an explicitly defined finite alias law.

The represented law is the integer column mass / (N * 2**precision).
Rounding thresholds preserves total mass by construction. Float alias-table
building and its distance to the supplied distribution remain diagnostics.
"""
import numpy as np


def build(probabilities, precision=44):
    probability=np.asarray(probabilities,dtype=np.float64)
    assert probability.ndim==1 and len(probability)>1 and np.all(np.isfinite(probability)) and np.all(probability>=0)
    probability=probability/probability.sum()
    n=1<<(len(probability)-1).bit_length()
    assert precision>=1 and precision+n.bit_length()-1<63
    scaled=np.pad(probability,(0,n-len(probability)))*n
    threshold=np.ones(n);alias=np.arange(n,dtype=np.int64)
    small=[int(i) for i in np.flatnonzero(scaled<1)]
    large=[int(i) for i in np.flatnonzero(scaled>=1)]
    while small and large:
        low,high=small.pop(),large.pop()
        threshold[low]=scaled[low];alias[low]=high
        scaled[high]=max(0.,scaled[high]+scaled[low]-1.)
        (small if scaled[high]<1 else large).append(high)
    denominator=1<<precision
    cut=np.rint(np.clip(threshold,0,1)*denominator).astype(np.int64)
    mass=cut.copy()
    np.add.at(mass,alias,denominator-cut)
    assert all(int(x)>=0 for x in mass)
    assert sum(int(x) for x in mass)==n*denominator
    represented=mass.astype(np.float64)/(n*denominator)
    target=np.pad(probability,(0,n-len(probability)))
    diagnostics=dict(vocabulary=len(probability),padded_size=n,precision=precision,
        threshold_bits=precision+1,integer_total_mass=n*denominator,
        observed_TV_to_supplied_distribution=float(.5*np.abs(represented-target).sum()),
        padded_mass=int(mass[len(probability):].sum()),zero_thresholds=int(np.count_nonzero(cut==0)),
        full_thresholds=int(np.count_nonzero(cut==denominator)))
    return cut,alias,mass,represented,diagnostics
