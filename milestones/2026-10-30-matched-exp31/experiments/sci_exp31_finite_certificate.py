"""Certify actual pinned SCI exp31 factors and a separate matched finite model.

Does not replace the earlier exp(c*delta) sampler profile. It defines a new
supplementary same-function dense/rejection comparison cell before execution.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rep(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))


def exp_enclosure(x):
    term=F(1);lower=term
    for k in range(1,129):term*=x/k;lower+=term
    upper=lower+(term*x/129)/(1-x/130)
    return lower,upper


def main():
    destination=ROOT/'results/sci_exp31_finite_certificate.json';assert not destination.exists()
    binary=ROOT/'toolchain/bounded-sampler/sci_exp31_factor_dump'
    actual=json.loads(subprocess.run([str(binary)],capture_output=True,text=True,check=True).stdout)
    assert [len(t) for t in actual['tables']]==[256,256,64]
    scale=1<<31;checked=0
    for digit,table in enumerate(actual['tables']):
        for value,observed in enumerate(table):
            lower,upper=exp_enclosure(F(value,1<<(20-8*digit)))
            lo=F(scale)/upper;hi=F(scale)/lower
            assert lo.numerator//lo.denominator==hi.numerator//hi.denominator==observed,(digit,value)
            checked+=1
    # Every F20 bounded score point has exactly represented nonnegative x=1-delta.
    y=np.arange((1<<21)+1,dtype=np.int64)
    tables=[np.array(t,dtype=np.int64) for t in actual['tables']]
    values=(tables[0][y&255]*tables[1][(y>>8)&255])>>31
    values=(values*tables[2][(y>>16)&63])>>31
    assert int(values.max())==scale and np.all(values>0)
    assert np.all(np.diff(values)<=0)
    # Three downward rounded factors and two downward rounded products:
    # absolute acceptance error <=5/S, independent of score and q.
    error=F(5,scale);minimum=F(1353,10000)
    _,e2upper=exp_enclosure(F(2))
    assert 1/e2upper-error>minimum
    model_tv=error/minimum
    failure=(1-minimum)**201
    cdf_tv=F(50304,1<<64)
    assert failure<F(1,1<<40) and cdf_tv<F(1,1<<40)
    # qmass sums to2^60. Its largest possible single entry is2^60,
    # so split low30/high31 rather than accidentally dropping endpoint bit60.
    assert (1<<60)>>(30)==1<<30
    out=dict(all_passed=True,public_factors_verified=checked,finite_score_grid_points=len(y),
        actual_public_factors=actual,profile=dict(score_fraction=20,exp_fraction=31,exp_output_bits=33,
            q_total_mass_bits=60,q_low_bits=30,q_high_bits=31,weight_low_product_bits=63,
            weight_high_product_bits=64,cdf_bits=92,cdf_uniform_bits=64,rejection_trials=201),
        acceptance_absolute_error_bound=rep(error),finite_model_tv_vs_ideal_q_exp_delta=rep(model_tv),
        rejection_failure_bound=rep(failure),dense_inverse_cdf_tv_bound=rep(cdf_tv),
        observed_min_acceptance_numerator=int(values.min()),observed_max_acceptance_numerator=int(values.max()),
        source_sha256={n:sha(ROOT/'experiments'/n) for n in ['sci_exp31_factor_dump.cpp','sci_exp31_finite_certificate.py']},
        binary_sha256=sha(binary),upstream_source_sha256=sha(ROOT/'third_party/FABLE/extern/EzPC/SCI/src/Math/math-functions.cpp'),
        limits='Supplementary matched finite model q_i*E31(delta_i), not silentreplacement of originalp44descending profile. Both futureCDF andfuturedirectBernoulli rejection must use the identical verified factors/order/floors. Samplingerror canremain<2^-40 relativethisfinite model; its additional approxTV~1.72e-8 versus idealqexp(delta) is separate and must be reported, in addition to previousscoreapprox/qcompilation/crypto failures. This is an analytical/table certificate only; neither completeCDF sampler nor matchedE31rejection has been implemented or measured yet. No newprimitive/CCFclaim.')
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,factors=checked,grid_points=len(y),model_tv=float(model_tv),
        rejection_failure=float(failure),cdf_tv=float(cdf_tv)),indent=2))


if __name__=='__main__':main()
