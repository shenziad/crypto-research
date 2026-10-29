"""Finite-grid approximation screen and independent integer domain checks."""
import json
import numpy as np
from pathlib import Path
from public_silu_spline import evaluate
from public_spline_fss import prepare,evaluate as fss_evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text())
    rng=np.random.default_rng(2026102922);rows=[];domain=[]
    grid=np.arange(-127*256,127*256+1,dtype=np.int64)
    for layer,spec in enumerate(data['specifications']):
        y=evaluate(spec,grid);physical=grid.astype(np.float64)*spec['sa']/256
        true=physical/(1+np.exp(-physical));error=y/(1<<spec['fraction_bits'])-true
        boundaries=np.unique(np.clip(np.asarray(spec['signed_starts'],dtype=np.int64)[:,None]+np.array([-1,0,1])[None,:],-(1<<31),(1<<31)-1))
        expected=[]
        for x in boundaries:
            segment=max(i for i,a in enumerate(spec['signed_starts']) if a<=int(x))
            expected.append(spec['slopes'][segment]*int(x)+spec['intercepts'][segment])
        assert np.array_equal(evaluate(spec,boundaries),np.asarray(expected,dtype=np.int64))
        assert max(abs(int(v)) for v in expected)<=spec['absolute_bound']
        rows.append(dict(layer=layer,exhaustive_grid_points=len(grid),max_abs_clipped_silu_grid_error=float(np.abs(error).max()),
            rmse=float(np.sqrt(np.mean(error**2))),boundary_points=len(boundaries),integer_boundary_checks_passed=True))
        if layer in (0,12,23):
            x=np.concatenate([boundaries,rng.integers(-(1<<31),1<<31,4096,dtype=np.int64)])
            keys,alpha=prepare(len(x));h=(x.view(np.uint64)+alpha)&np.uint64((1<<32)-1)
            output=fss_evaluate(keys[0],h,spec,0)+fss_evaluate(keys[1],h,spec,1)
            assert np.array_equal(output.view(np.int64),evaluate(spec,x))
            domain.append(dict(layer=layer,full_signed32_boundary_random_inputs=len(x),all_passed=True))
    result=dict(layers=rows,native_full32_domain_checks=domain,all_passed=True,
        limits='Grid errors use float64 SiLU reference, not rigorous real-arithmetic error intervals or a downstream model theorem. Piece endpoint range claims use exact integer coefficients. Native full32 checks are functional only.')
    (ROOT/'results/public_silu_spline_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('all24 grid points',sum(r['exhaustive_grid_points'] for r in rows),'max grid error',max(r['max_abs_clipped_silu_grid_error'] for r in rows),'domain',domain)


if __name__=='__main__':main()
