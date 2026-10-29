"""Check sparse DCF evaluator per-party and real32 quantized-gate identity."""
import hashlib
import json
import time
import numpy as np
from fused_gated_lut import ROOT,random64,split64
from dcf_aes_native import generate_batch,evaluate_batch
from dcf_sparse_native import derivative,weighted,public_constant,LIB


def main():
    rng=np.random.default_rng(2026102917);table=rng.integers(-100,101,256,dtype=np.int64)
    thresholds,deltas,f0=derivative(table,8,3,3);weights=(-deltas).view(np.uint64)
    alpha=np.arange(256,dtype=np.uint64);beta=random64((256,2));keys=generate_batch(alpha,beta,8,64)
    pairs=0
    for h in range(256):
        opened=np.full(256,h,dtype=np.uint64)
        for p in (0,1):
            direct=np.zeros((256,2),dtype=np.uint64)
            for t,w in zip(thresholds,weights):
                query=(opened-t)&np.uint64(255)
                direct+=w*evaluate_batch(keys[p],query,p,8,2,64)
            for cached in (False,True):assert np.array_equal(weighted(keys[p],opened,thresholds,weights,p,8,2,cached),direct)
        pairs+=256
    print('individual-party sparse/tree outputs checked',pairs,flush=True)
    capture=ROOT/'results/qwen_gate_fixed8_input_late_capture.npz';rows=[]
    with np.load(capture) as data:
        for layer in (0,12,23):
            prefix=f'layer_layer_{layer}';a=data[prefix+'_a_fixed32'].view(np.uint64)&np.uint64((1<<32)-1)
            b=data[prefix+'_b_integer'].view(np.uint64);table=data[prefix+'_lookup'];n=len(a)
            thresholds,deltas,f0=derivative(table);weights=(-deltas).view(np.uint64)
            mask=random64((n,))&np.uint64((1<<32)-1);rb=random64((n,));beta=np.stack([np.ones(n,dtype=np.uint64),-rb],axis=1)
            keys=generate_batch(mask,beta,32,64);bs=split64(beta)
            opened=(a+mask)&np.uint64((1<<32)-1);constant=public_constant(opened,thresholds,deltas,f0)
            qa=np.clip((((a^np.uint64(1<<31)).astype(np.int64)-(1<<31))+128)//256,-127,127)
            assert np.array_equal((qa&255).astype(np.uint8),data[prefix+'_a_index'])
            measurements=[]
            for repetition in range(3):
                order=(False,True) if repetition%2==0 else (True,False)
                reference=None
                for cached in order:
                    start=time.perf_counter_ns()
                    values=[weighted(keys[p],opened,thresholds,weights,p,cached=cached)+bs[p]*constant[:,None] for p in (0,1)]
                    ms=(time.perf_counter_ns()-start)/1e6
                    for p in (0,1):
                        if reference is not None:assert np.array_equal(values[p],reference[p])
                    reference=values
                    actual=values[0]+values[1]
                    assert np.array_equal(actual,beta*table[qa&255].view(np.uint64)[:,None])
                    product=(b+rb)*actual[:,0]+actual[:,1]
                    assert np.array_equal(product,b*table[qa&255].view(np.uint64))
                    measurements.append(dict(cached_public_prefix=cached,repetition=repetition,both_party_local_ms=ms))
                print('layer',layer,'repeat',repetition,'nonzero jumps',len(thresholds),flush=True)
            rows.append(dict(layer=layer,n=n,nonzero_jumps=len(thresholds),all_passed=True,
                core_dcf_key_bytes_both_parties=sum(k.nbytes for k in keys),measurements=measurements))
    out=dict(native_individual_party_pairs=pairs,real_layers=rows,all_passed=True,native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        limits='Local function-evaluation benchmark, not a two-party protocol timing. Same key/input reused only for repeated identical implementation measurements, never a new secret masked opening. B quantization/clipping and private projection excluded. Standard DCF closure; independent novelty unconfirmed.')
    (ROOT/'results/dcf_sparse_native_check.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:
        for cached in (False,True):
            values=[v['both_party_local_ms'] for v in r['measurements'] if v['cached_public_prefix']==cached]
            print(r['layer'],'cached',cached,'median both-party local ms',np.median(values),flush=True)


if __name__=='__main__':main()
