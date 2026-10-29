"""Small full-domain exact classifier/body check, including independent raw paths."""
import json
import numpy as np
from fused_gated_lut import ROOT
from exact_coarse_spline import prepare,masked_classifier,evaluate,raw
from dcf_wide_native import select
from dithered_spline_fss import parameters,classifier_bits


def direct(key,h,t,spec,p,bits,variant):
    s,e,m,c=parameters(spec,bits);view=classifier_bits(spec);dim=4 if variant=='weighted' else 2;out=np.zeros((len(h),2),dtype=np.uint64)
    with np.errstate(over='ignore'):
        for i,st in enumerate(s):
            old=(i-1)%len(s);dm=m[old]-m[i];dc=c[old]-c[i]
            values=select(key,(t-st-np.uint64(1))&np.uint64((1<<view)-1),np.arange(dim,dtype=np.int32),p,view,dim)
            for j in range(2 if variant=='weighted' else 1):out[:,j]+=(dm*h+dc)*values[:,2*j]-dm*values[:,2*j+1]
    return out


def toy_spec():
    starts=[-32,-8,0,8];ends=[-8,0,8,32];slopes=[3,-2,4,1];intercepts=[7,11,-3,5];pieces=[]
    for a,b,m,c in zip(starts,ends,slopes,intercepts):
        if a<0:pieces.append((a+64,min(b,0)+64,m,c-m*64))
        if b>0:pieces.append((max(a,0),b,m,c))
    pieces.sort()
    return dict(input_bits=6,classification_shift=2,unsigned_pieces=pieces,use_dither_fss=True),starts,slopes,intercepts


def main():
    spec,starts,slopes,intercepts=toy_spec();alpha=np.repeat(np.arange(4096,dtype=np.uint64),16);gamma=np.tile(np.arange(16,dtype=np.uint64),4096);rows=[]
    for variant in ('basis','weighted'):
        keys,_,_=prepare(len(alpha),spec,12,variant,alpha,gamma);pairs=0
        for x in range(-32,32):
            h=(alpha+np.uint64(x%4096))&np.uint64(4095)
            ts=[masked_classifier(keys[p],h,spec,p) for p in (0,1)];t=(ts[0]+ts[1])&np.uint64(15)
            assert np.array_equal(t,(gamma+np.uint64((x//4)%16))&np.uint64(15))
            for p in (0,1):
                # A separately evaluated per-query reference on a subbatch.
                n=1024;k=keys[p]['dcf'][:n]
                assert np.array_equal(raw(k,h[:n],t[:n],spec,p,12,variant),direct(k,h[:n],t[:n],spec,p,12,variant))
            parts=[evaluate(keys[p],h,t,spec,p,12,variant) for p in (0,1)];total=(parts[0]+parts[1])&np.uint64(4095)
            i=int(np.searchsorted(starts[1:],x,side='right'));expected=np.uint64((slopes[i]*x+intercepts[i])%4096)
            if variant=='weighted':
                rb=(keys[0]['b_mask']+keys[1]['b_mask'])&np.uint64(4095)
                assert np.all(total[:,0]==expected) and np.array_equal(total[:,1],(-rb*expected)&np.uint64(4095))
            else:assert np.all(total==expected)
            pairs+=len(alpha)
        rows.append(dict(variant=variant,pairs=pairs,all_body_masks=True,all_independent_coarse_masks=True,all_signed6_inputs=True,discontinuous_polynomials=True,passed=True));print(rows[-1],flush=True)
    (ROOT/'results/exact_coarse_spline_check.json').write_text(json.dumps(dict(records=rows,all_passed=True,
        limits='Exact aligned function; noC0 smoothness needed. One2-bitlowDCF plus4-bitcoarseDCF,12-bitbody. Trusted local reference; no timing, complete security or novelty claim.'),indent=2)+'\n')


if __name__=='__main__':main()
