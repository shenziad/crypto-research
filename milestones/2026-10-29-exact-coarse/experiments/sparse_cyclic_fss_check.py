"""Constructive screen: public step function from one hidden-offset DCF.

This is a linear postprocessing identity using standard DCF, not an established
new primitive. All tests local, mathematical/correctness only.
"""
import json
import time
import numpy as np
from fused_gated_lut import ROOT,random64,split64
from dcf_aes_native import generate_batch,evaluate_batch


def main():
    bits,shift,maximum=8,3,3;ring=1<<bits
    x=np.arange(ring,dtype=np.int64);signed=(x^(ring//2))-ring//2
    q=np.clip((signed+(1<<(shift-1)))//(1<<shift),-maximum,maximum)
    # Deliberately nonmonotone public finite gate, including nonzero F(0).
    table=np.array([13,-9,27,4,-31,8,19],dtype=np.int64)
    f=table[q+maximum]
    thresholds=np.nonzero(f[1:]!=f[:-1])[0]+1;deltas=f[thresholds]-f[thresholds-1]
    assert f[-1]==f[0] and deltas.sum()==0
    alpha=np.arange(ring,dtype=np.uint64)
    beta=np.stack([np.ones(ring,dtype=np.uint64),-random64((ring,))],axis=1)
    keys=generate_batch(alpha,beta,bits,64);bs=split64(beta)
    started=time.perf_counter();pairs=0
    for h in range(ring):
        constant=int(f[0])-sum(int(d) for t,d in zip(thresholds,deltas) if h<int(t))
        accum=[bs[p]*np.uint64(constant%(1<<64)) for p in (0,1)]
        for threshold,delta in zip(thresholds,deltas):
            query=np.full(ring,(h-int(threshold))%ring,dtype=np.uint64)
            for p in (0,1):
                accum[p]-=np.uint64(int(delta)%(1<<64))*evaluate_batch(keys[p],query,p,bits,2,64)
        expected=beta*f[(h-alpha.astype(np.int64))%ring].view(np.uint64)[:,None]
        assert np.array_equal(accum[0]+accum[1],expected);pairs+=ring
    # Quantized/clipped gate input32, removed8 bits, symmetric127: 254 internal
    # code transitions plus signed-boundary jump (at most255 nonzero jumps).
    result=dict(input_bits=bits,rounding_removed_bits=shift,clip_bound=maximum,
        nonzero_public_derivative_points=len(thresholds),native_threshold_input_pairs=pairs,
        output_dimension=2,all_passed=True,seconds=time.perf_counter()-started,
        identity='F(h-alpha)=F(last)-sum_T delta_T*I(h<T)+(sum_T delta_T)*I(h<alpha)-sum_T delta_T*I((h-T)modR<alpha)',
        llm_candidate='Apply sparse public derivative of quantize+clip+SiLU LUT directly to wide masked input; combine vector payload(1,-rb) for gated product.',
        limits='Sparse cyclic convolution is a standard linear closure consequence of DCF; independent novelty not established. No full32 benchmark, private B path, GPU protocol or end-to-end gain yet. More local DCF evaluations can outweigh saved material/rounds.')
    (ROOT/'results/sparse_cyclic_fss_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
