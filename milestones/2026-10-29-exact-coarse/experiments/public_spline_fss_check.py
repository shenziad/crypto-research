"""Scalar-basis FSS per-party native checks and real high-precision gates."""
import hashlib
import json
import time
import numpy as np
from fused_gated_lut import ROOT
from dcf_aes_native import evaluate_batch
from public_silu_spline import evaluate as plain_evaluate
from public_spline_fss import prepare,queries,linear,evaluate,LIB


def toy_spec(f):
    d=np.diff(f);starts=np.r_[0,np.nonzero(d[1:]!=d[:-1])[0]+1];ends=np.r_[starts[1:],len(d)]
    return dict(delta_runs=[(int(a+1),int(b),int(d[a])) for a,b in zip(starts,ends) if int(d[a])],function_at_last=int(f[-1]))


def main():
    ring=256;x=np.arange(ring,dtype=np.int64);s=(x^128)-128
    f=np.where(s<-64,-13,np.where(s<-16,2*s+115,np.where(s<24,-3*s+27,np.where(s<64,5*s-163,157))))
    spec=toy_spec(f);keys,alpha=prepare(ring,8,np.arange(ring,dtype=np.uint64));pairs=0
    for h in range(ring):
        opened=np.full(ring,h,dtype=np.uint64);compiled=queries(spec,opened,8);q,w,_,_=compiled
        values=[]
        for p in (0,1):
            direct=np.zeros(ring,dtype=np.uint64)
            for j in range(q.shape[1]):
                dcf=evaluate_batch(keys[p]['dcf'],q[:,j],p,8,2,64)
                direct+=(dcf*w[:,j,:]).sum(axis=1,dtype=np.uint64)
            for cached in (False,True):assert np.array_equal(linear(keys[p]['dcf'],q,w,p,8,cached),direct)
            values.append(evaluate(keys[p],opened,spec,p,8,compiled=compiled))
        assert np.array_equal(values[0]+values[1],f[(h-alpha.astype(np.int64))%ring].view(np.uint64));pairs+=ring
    constant=toy_spec(np.full(ring,-17,dtype=np.int64));h=np.arange(ring,dtype=np.uint64)
    assert np.all((evaluate(keys[0],h,constant,0,8)+evaluate(keys[1],h,constant,1,8)).view(np.int64)==-17)
    print('all small-ring native and scalar identity pairs',pairs,flush=True)
    quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());rows=[]
    capture=ROOT/'results/qwen_public_spline_f22_capture.npz'
    with np.load(capture) as data:
        for layer in (0,12,23):
            a=data[f'layer_{layer}_a_fixed32'].astype(np.int64);n=len(a);spec=quality['specifications'][layer]
            assert np.array_equal(plain_evaluate(spec,a),data[f'layer_{layer}_gate_numerator'])
            keys,alpha=prepare(n);opened=(a.view(np.uint64)+alpha)&np.uint64((1<<32)-1)
            t0=time.perf_counter_ns();compiled=queries(spec,opened);compile_ms=(time.perf_counter_ns()-t0)/1e6
            measurements=[]
            for repeat in range(3):
                reference=None
                for cached in ((False,True) if repeat%2==0 else (True,False)):
                    start=time.perf_counter_ns();out=[evaluate(keys[p],opened,spec,p,cached=cached,compiled=compiled) for p in (0,1)]
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    if reference is not None:
                        for p in (0,1):assert np.array_equal(out[p],reference[p])
                    reference=out
                    assert np.array_equal((out[0]+out[1]).view(np.int64),data[f'layer_{layer}_gate_numerator'])
                    measurements.append(dict(cached=cached,both_party_sequential_ms=elapsed,repeat=repeat))
            key_bytes=sum(v.nbytes for k in keys for v in k.values())
            rows.append(dict(layer=layer,n=n,public_pieces=len(spec['slopes']),public_delta_runs=len(spec['delta_runs']),
                public_queries_per_input=compiled[0].shape[1],compile_public_queries_ms=compile_ms,
                total_key_bytes_both_parties=key_bytes,all_passed=True,measurements=measurements))
            print('layer',layer,'pieces',len(spec['slopes']),'queries',compiled[0].shape[1],'key bytes',key_bytes,
                'cached ms',np.median([r['both_party_sequential_ms'] for r in measurements if r['cached']]),flush=True)
    result=dict(small_ring_mask_input_pairs=pairs,constant_function_checked=True,real_layers=rows,all_passed=True,
        native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        limits='Functional FSS kernel and local sequential evaluation only, not full protocol timing. Public queries compiled once per fixed public opening. Repeated same key/input for implementation timing only; no different secret opening reuses masks. Public-basis specialization novelty and full joint-view proof pending.')
    (ROOT/'results/public_spline_fss_check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
