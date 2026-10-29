"""Reference interval aggregation and piecewise-affine offset evaluation."""
import json
import numpy as np
from dcf_public_range import prepare,interval_mass
from dcf_aes_native import evaluate_batch
from fused_gated_lut import ROOT,random64,split64


def main():
    bits=8;ring=1<<bits;alpha=np.arange(ring,dtype=np.uint64);n=ring
    beta=np.stack([np.ones(n,dtype=np.uint64),-random64((n,))],axis=1)
    keys=prepare(alpha,beta,bits);beta_shares=split64(beta);interval_pairs=0
    for left in range(ring):
        for length in (1,2,127,128,255,256):
            right=(left+length-1)%ring
            l=np.full(n,left,dtype=np.uint64);r=np.full(n,right,dtype=np.uint64)
            values=[interval_mass(keys[p],l,r,p,bits) for p in (0,1)]
            if left<=right:count=np.maximum(np.minimum(alpha,right+1).astype(np.int64)-np.minimum(alpha,left).astype(np.int64),0)
            else:count=alpha-np.minimum(alpha,left)+np.minimum(alpha,right+1)
            assert np.array_equal(values[0]+values[1],beta*count.astype(np.uint64)[:,None]);interval_pairs+=n
    # Bounded integer piecewise-affine function with jumps and signed branches.
    z=np.arange(ring,dtype=np.int64);x=(z^(ring//2))-ring//2
    f=np.where(x<-64,-13,np.where(x<-16,2*x+115,np.where(x<24,-3*x+27,np.where(x<64,5*x-163,157))))
    d=np.diff(f);starts=np.r_[0,np.nonzero(d[1:]!=d[:-1])[0]+1];ends=np.r_[starts[1:],len(d)]
    runs=[(int(a+1),int(b),int(d[a])) for a,b in zip(starts,ends) if int(d[a])!=0]
    total_delta=int(f[-1]-f[0]);pairs=0
    for h in range(ring):
        opened=np.full(n,h,dtype=np.uint64)
        public_sum=sum(slope*max(0,b-max(a,h+1)+1) for a,b,slope in runs)
        values=[beta_shares[p]*np.uint64((int(f[-1])-public_sum)%(1<<64)) for p in (0,1)]
        for p in (0,1):values[p]+=np.uint64(total_delta%(1<<64))*evaluate_batch(keys[p]['beta'],opened,p,bits,2,64)
        for a,b,slope in runs:
            left=np.full(n,(h-b)%ring,dtype=np.uint64);right=np.full(n,(h-a)%ring,dtype=np.uint64)
            for p in (0,1):values[p]-=np.uint64(slope%(1<<64))*interval_mass(keys[p],left,right,p,bits)
        expected=beta*f[(h-alpha.astype(np.int64))%ring].view(np.uint64)[:,None]
        assert np.array_equal(values[0]+values[1],expected);pairs+=n
    out=dict(interval_threshold_pairs=interval_pairs,piecewise_affine_mask_input_pairs=pairs,
        nonzero_discrete_slope_runs=len(runs),all_passed=True,
        prior='Existing public interval/spline offset FSS, EUROCRYPT2021; this is a reference implementation for strong baselines.',
        limits='Native functional identity checks only. Two independent vector2 DCF keys, no merged vector4 optimization. No model or secure full-block performance claim.')
    (ROOT/'results/dcf_public_range_check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)


if __name__=='__main__':main()
