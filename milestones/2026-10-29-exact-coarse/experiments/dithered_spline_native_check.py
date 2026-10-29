"""Independent raw endpoint checks and full-width real-vector reconstruction."""
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT,random64
from dcf_wide_native import select
from dithered_spline_fss import prepare,evaluate,raw,parameters,public_classifier,classifier_bits,LIB
from dithered_silu import build,evaluate as plain


def reconstruct(keys,h,spec,bits,variant):
    mask=np.uint64((1<<bits)-1);values=[evaluate(keys[p],h,spec,p,bits,variant) for p in (0,1)]
    if variant=='selector':
        sh=(values[0][:,0]+keys[0]['horner_rs']+values[1][:,0]+keys[1]['horner_rs'])&mask;out=[]
        for p in (0,1):
            g=keys[p]['horner_product']-h*keys[p]['horner_rs']-sh*keys[p]['input_mask']+values[p][:,1]+keys[p]['refresh']
            if p==0:g+=h*sh
            out.append(g)
        return (out[0]+out[1])&mask,None
    total=(values[0]+values[1])&mask
    return (total[:,0],total[:,1]) if variant=='weighted' else (total,None)


def direct(key,h,spec,party,bits,variant):
    s,e,m,c=parameters(spec,bits);hc=public_classifier(h,spec);view=classifier_bits(spec);mask=np.uint64((1<<view)-1)
    dim=1 if variant=='selector' else 2 if variant=='basis' else 4 if variant=='weighted' else 2*len(s)
    out=np.zeros((len(h),2),dtype=np.uint64)
    with np.errstate(over='ignore'):
        for i,st in enumerate(s):
            previous=(i-1)%len(s);dm=m[previous]-m[i];dc=c[previous]-c[i];q=(hc-st-np.uint64(1))&mask
            co=np.asarray([2*previous,2*previous+1,2*i,2*i+1] if variant=='wide' else np.arange(dim),dtype=np.int32)
            d=select(key,q,co,party,view,dim)
            if variant=='selector':out[:,0]+=dm*d[:,0];out[:,1]+=dc*d[:,0]
            elif variant=='wide':out[:,0]+=h*(d[:,0]-d[:,2])+d[:,1]-d[:,3]
            else:
                for j in range(2 if variant=='weighted' else 1):out[:,j]+=(dm*h+dc)*d[:,2*j]-dm*d[:,2*j+1]
    return out


def main():
    starts=[-128,-40,-8,0,20,48];ends=starts[1:]+[128];slopes=[0,2,-1,3,1,0];intercepts=[3,83,59,59,99,147];pieces=[]
    for a,b,m,c in zip(starts,ends,slopes,intercepts):
        if a<0:pieces.append((a+256,min(b,0)+256,m,c-m*256))
        if b>0:pieces.append((max(a,0),b,m,c))
    pieces.sort();spec=dict(input_bits=8,classification_shift=2,unsigned_pieces=pieces)
    alpha=(np.arange(256,dtype=np.uint64)*np.uint64(17))&np.uint64(4095);records=[]
    for variant in ('selector','basis','weighted','wide'):
        keys,_=prepare(len(alpha),spec,12,alpha,variant);pairs=0
        for x in range(-64,64):
            h=(alpha+np.uint64(x%4096))&np.uint64(4095)
            for p in (0,1):assert np.array_equal(raw(keys[p]['dcf'],h,spec,p,12,variant),direct(keys[p]['dcf'],h,spec,p,12,variant))
            rho=(h&np.uint64(3)).astype(np.int64);z=(x//4+(rho<x%4).astype(np.int64))*4
            i=np.searchsorted(starts[1:],z,side='right');target=(np.asarray(slopes,dtype=np.int64)[i]*x+np.asarray(intercepts,dtype=np.int64)[i]).view(np.uint64)&np.uint64(4095)
            result,aux=reconstruct(keys,h,spec,12,variant);assert np.array_equal(result,target)
            if aux is not None:
                rb=(keys[0]['b_mask']+keys[1]['b_mask'])&np.uint64(4095);assert np.array_equal(aux,(-rb*target)&np.uint64(4095))
            pairs+=len(alpha)
        records.append(dict(variant=variant,toy_pairs=pairs,individual_party_raw_equal=True));print(records[-1],flush=True)
    quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());real=[];mask=np.uint64((1<<63)-1)
    with np.load(ROOT/'results/qwen_public_spline_f22_capture.npz') as data:
        for layer in (0,12,23):
            a=data[f'layer_{layer}_a_fixed32'];spec=build(quality['specifications'][layer]['sa'],8)
            for variant in ('selector','basis','weighted','wide'):
                keys,alpha=prepare(len(a),spec,63,variant=variant);h=(a.view(np.uint64)+alpha)&mask
                expected=plain(spec,a,(h&np.uint64(255)).astype(np.int64)).view(np.uint64)&mask
                result,aux=reconstruct(keys,h,spec,63,variant);assert np.array_equal(result,expected)
                if aux is not None:
                    rb=(keys[0]['b_mask']+keys[1]['b_mask'])&mask;assert np.array_equal(aux,(-rb*expected)&mask)
                real.append(dict(layer=layer,variant=variant,n=len(a),classifier_bits=25,body_bits=63,passed=True))
            print('real layer',layer,'passed',flush=True)
    # The full-width reference computes the SAME randomized function: shift
    # its public classification input by B-1-rho, keeping the polynomial body.
    # Its preprocessing classifier mask is unshifted alpha, not public_classifier(alpha).
    pairs=0
    for variant in ('selector','basis','weighted','wide'):
        spec=dict(input_bits=8,classification_shift=2,unsigned_pieces=pieces,full_classifier_reference=True)
        keys,_=prepare(len(alpha:=np.arange(256,dtype=np.uint64)*np.uint64(17)&np.uint64(4095)),spec,12,alpha,variant)
        for x in range(-64,64):
            h=(alpha+np.uint64(x%4096))&np.uint64(4095);rho=(h&np.uint64(3)).astype(np.int64)
            for p in (0,1):assert np.array_equal(raw(keys[p]['dcf'],h,spec,p,12,variant),direct(keys[p]['dcf'],h,spec,p,12,variant))
            z=(x//4+(rho<x%4).astype(np.int64))*4;i=np.searchsorted(starts[1:],z,side='right')
            target=(np.asarray(slopes,dtype=np.int64)[i]*x+np.asarray(intercepts,dtype=np.int64)[i]).view(np.uint64)&np.uint64(4095)
            result,_=reconstruct(keys,h,spec,12,variant);assert np.array_equal(result,target);pairs+=len(alpha)
    out=dict(toy_records=records,real_records=real,full_width_same_randomized_function_pairs=pairs,all_passed=True,native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Matched optimized local FSS kernel functional checks. Public-dither randomized function differs from exact spline; no full-block speed, fullMPC security or novelty claim.')
    (ROOT/'results/dithered_spline_native_check.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
