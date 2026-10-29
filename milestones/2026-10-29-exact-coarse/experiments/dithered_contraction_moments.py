"""Exact per-coordinate dither moments; downstream linear MSE Monte Carlo check.

Uses held-out captured vectors to verify a formula, never to select parameters.
"""
import json
from pathlib import Path
import numpy as np
from dithered_silu import evaluate

ROOT=Path(__file__).resolve().parents[1]


def coordinate_errors(spec,a,shift,coarse_body=False):
    bucket=1<<shift;a=np.asarray(a,dtype=np.int64);r=a%bucket;lo=(a//bucket)*bucket;hi=lo+bucket
    target=evaluate(spec,a)
    if coarse_body:e0=evaluate(spec,lo)-target;e1=evaluate(spec,hi)-target
    else:
        idx=np.searchsorted(np.asarray(spec['signed_starts'][1:]),hi,side='right')
        e0=np.zeros_like(a);e1=np.asarray(spec['slopes'])[idx]*a+np.asarray(spec['intercepts'])[idx]-target
    numerator=(bucket-r)*e0+r*e1
    if coarse_body:assert np.all(numerator==0),'C0 aligned interpolation is unbiased at fixed input'
    p=r/bucket;mu=numerator/bucket;variance=p*(1-p)*(e1-e0).astype(np.float64)**2
    return e0,e1,p,mu,variance


def main():
    source=json.loads((ROOT/'results/qwen_public_spline_quality_f22_alignedb8.json').read_text())
    rng=np.random.default_rng(20261029);rows=[];samples=1024
    with np.load(ROOT/'results/qwen_public_spline_f22_alignedb8_capture.npz') as cap,np.load(ROOT/'results/qwen_down_int12_selected.npz') as weights:
        for layer in (0,12,23):
            spec=source['specifications'][layer];a=cap[f'layer_{layer}_a_fixed32'];b=cap[f'layer_{layer}_b_integer'].astype(np.float64)
            w=weights[f'layer_{layer}'].astype(np.float64)
            # Normalize before dot products. This is real moment arithmetic,
            # distinct from exact integer protocol accumulators and final floor.
            operator=w*(b[None,:]/(1<<(spec['fraction_bits']+12)))
            norm2=(operator*operator).sum(0)
            for shift in (4,6,8):
                for coarse in (False,True):
                    e0,e1,p,mu,var=coordinate_errors(spec,a,shift,coarse)
                    bias=operator@mu;bias2=float(bias@bias);variance=float(norm2@var);pred=bias2+variance
                    rho=rng.integers(0,1<<shift,(samples,len(a)))
                    err=np.where(rho<(a%(1<<shift))[None,:],e1[None,:],e0[None,:]).astype(np.float64)
                    empirical=err@operator.T;sq=(empirical*empirical).sum(1);mean=float(sq.mean());se=float(sq.std(ddof=1)/np.sqrt(samples))
                    assert abs(mean-pred)<=6*se+1e-15
                    row=dict(layer=layer,shift=shift,coarse_body=coarse,predicted_bias_norm_squared=bias2,
                        predicted_variance_trace=variance,predicted_mse=pred,empirical_mse=mean,empirical_standard_error=se,
                        monte_carlo_samples=samples,check_within_six_standard_errors=True,
                        maximum_absolute_coordinate_mean=float(np.max(np.abs(mu))))
                    rows.append(row);print(layer,shift,'coarse' if coarse else 'fine','predMSE',pred,'observed',mean,'SE',se,flush=True)
    (ROOT/'results/dithered_contraction_moments.json').write_text(json.dumps(dict(records=rows,all_passed=True,
        interpretation='Same alignedb8 finite function at every shift; only classifier resolution changes. For independent percoordinate publicrho, E||WdiagB error||² equals biasnorm² plus weighted variance trace.',
        limits='Three heldout firsttoken vectors used for mathematical verification, not calibration/parameter selection. Integer percoordinate means are exact rational formulas; downstream real moments and Monte Carlo useFP64. Before final floor and without physicalsb factor; no modelNLL or private protocol claim.'),indent=2)+'\n')


if __name__=='__main__':main()
