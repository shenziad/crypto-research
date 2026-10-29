"""Public interval sum of a hidden-threshold DCF using two vector DCFs.

Standard public spline/interval FSS technique; EUROCRYPT2021 is direct prior.
This version keeps independent two-coordinate keys instead of merging four
payload coordinates. It is a constructive reference, not a new primitive.
"""
import numpy as np
from dcf_aes_native import generate_batch,evaluate_batch
from fused_gated_lut import split64


def prepare(alpha,beta,bits):
    alpha=np.asarray(alpha,dtype=np.uint64);beta=np.asarray(beta,dtype=np.uint64)
    assert beta.shape==(len(alpha),2)
    ab=alpha[:,None]*beta;shares=split64(ab)
    kb=generate_batch(alpha,beta,bits,64);ka=generate_batch(alpha,ab,bits,64)
    return [dict(beta=kb[p],alpha_beta=ka[p],alpha_beta_share=shares[p]) for p in (0,1)]


def interval_mass(key,left,right,party,bits):
    """Shares of beta * sum_{u in cyclic[left,right]} I(u<alpha)."""
    ring=1<<bits;left=np.asarray(left,dtype=np.uint64);right=np.asarray(right,dtype=np.uint64)
    previous=(left-np.uint64(1))&np.uint64(ring-1)
    def prefix(v):
        b=evaluate_batch(key['beta'],v,party,bits,2,64)
        ab=evaluate_batch(key['alpha_beta'],v,party,bits,2,64)
        return key['alpha_beta_share']+(v+np.uint64(1))[:,None]*b-ab
    wrap=((left==0)|(left>right)).astype(np.uint64)
    return prefix(right)-prefix(previous)+wrap[:,None]*key['alpha_beta_share']
