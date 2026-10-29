"""Exhaustive separated classifier/body native DCF reconstruction checks."""
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from public_polynomial_basis import prepare,evaluate
from dcf_wide_native import LIB


def main():
    starts_signed=[-128,-40,-8,0,20,48];ends_signed=starts_signed[1:]+[128]
    slopes=[0,2,-1,3,1,0];intercepts=[3,83,59,59,99,147]
    pieces=[]
    for a,b,m,c in zip(starts_signed,ends_signed,slopes,intercepts):
        if a<0:pieces.append((a+256,min(b,0)+256,m,c))
        if b>0:pieces.append((max(a,0),b,m,c))
    pieces.sort();starts=[a//4 for a,_,_,_ in pieces];coefficients=[[c,m] for _,_,m,c in pieces]
    alpha=np.arange(4096,dtype=np.uint64);ca=(alpha&np.uint64(255))>>np.uint64(2);rng=np.random.default_rng(20261029)
    max_error=0;max_mean_error=0;records=[]
    for weighted in (False,True):
        beta=rng.integers(0,4096,len(alpha),dtype=np.uint64) if weighted else np.ones(len(alpha),dtype=np.uint64)
        keys=prepare(starts,coefficients,alpha,beta,6,12,classifier_alpha=ca);pairs=0
        for x in range(-64,64):
            h=(alpha+np.uint64(x%4096))&np.uint64(4095);hc=(h&np.uint64(255))>>np.uint64(2);rho=(h&np.uint64(3)).astype(np.int64)
            coarse=(x//4+(rho<(x%4)).astype(np.int64))*4
            index=np.searchsorted(starts_signed[1:],coarse,side='right')
            target=np.asarray(slopes,dtype=np.int64)[index]*x+np.asarray(intercepts,dtype=np.int64)[index]
            expected=(target.view(np.uint64)*beta)&np.uint64(4095)
            values=[evaluate(keys[p],h,starts,coefficients,p,6,12,classifier_opened=hc) for p in (0,1)]
            assert np.array_equal((values[0]+values[1])&np.uint64(4095),expected),(weighted,x)
            exact_index=np.searchsorted(starts_signed[1:],x,side='right');exact=slopes[exact_index]*x+intercepts[exact_index]
            error=target-exact;max_error=max(max_error,int(np.abs(error).max()));max_mean_error=max(max_mean_error,abs(float(error.mean())))
            assert np.array_equal(np.bincount(rho,minlength=4),np.full(4,1024))
            pairs+=len(alpha)
        records.append(dict(weighted=weighted,body_bits=12,guarded_view_bits=8,classifier_bits=6,promised_input_bits=7,all_body_masks=4096,input_count=128,pairs=pairs,passed=True))
        print(records[-1],flush=True)
    assert max_error<=max(abs(slopes[i]-slopes[i-1])*3 for i in range(1,len(slopes)))
    result=dict(records=records,all_passed=True,total_pairs=sum(v['pairs'] for v in records),maximum_integer_error=max_error,maximum_expected_bias=max_mean_error,
        native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Generic native DCF reference, not an optimized new kernel. Public rho is part of the randomized ideal functionality; no private-dither claim. Separate privacy proof, real numerical quality and novelty still required.')
    (ROOT/'results/dithered_classifier_reference_check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
