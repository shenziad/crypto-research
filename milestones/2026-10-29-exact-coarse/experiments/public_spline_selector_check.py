"""Independent local query-path checks for selector and weighted kernels."""
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from dcf_aes_native import evaluate_batch
from dcf_wide_native import select
from public_spline_coefficient_basis import parameters
from public_spline_selector import prepare,native,evaluate,LIB


def main():
    pieces=[(0,17,2,3),(17,64,-1,47),(64,128,0,29),(128,193,0,-41),(193,256,3,-769)]
    spec=dict(unsigned_pieces=pieces,input_bits=8);starts,ends,m,c=parameters(spec,8)
    f=np.asarray([next(v*x+w for a,b,v,w in pieces if a<=x<b) for x in range(256)],dtype=np.int64)
    alpha=np.arange(256,dtype=np.uint64);n=len(alpha);all_keys={v:prepare(n,spec,8,alpha,weighted=v)[0] for v in (False,True)}
    pairs=0
    for h in range(256):
        opened=np.full(n,h,dtype=np.uint64)
        for weighted in (False,True):
            keys=all_keys[weighted]
            for p in (0,1):
                direct=np.zeros((n,2),dtype=np.uint64)
                for i,start in enumerate(starts):
                    prev=(i+len(starts)-1)%len(starts);dm=m[prev]-m[i];dc=c[prev]-c[i]
                    q=(opened-start-np.uint64(1))&np.uint64(255)
                    if weighted:
                        d=select(keys[p]['dcf'],q,np.arange(4),p,8,4)
                        w0=dm*opened+dc;w1=-dm
                        for j in (0,1):direct[:,j]+=w0*d[:,2*j]+w1*d[:,2*j+1]
                    else:
                        d=evaluate_batch(keys[p]['dcf'],q,p,8,1,64)[:,0]
                        direct[:,0]+=dm*d;direct[:,1]+=dc*d
                assert np.array_equal(native(keys[p]['dcf'],opened,spec,p,8,weighted),direct)
            values=[evaluate(keys[p],opened,spec,p,8,weighted) for p in (0,1)]
            if weighted:
                actual=(values[0]+values[1])&np.uint64(255);target=f[(h-alpha.astype(np.int64))%256].view(np.uint64)
                rb=(keys[0]['b_mask']+keys[1]['b_mask'])&np.uint64(255)
                assert np.array_equal(actual[:,0],target&np.uint64(255))
                assert np.array_equal(actual[:,1],(-rb*target)&np.uint64(255))
            else:
                sm=[(values[p][:,0]+keys[p]['horner_rs'])&np.uint64(255) for p in (0,1)];sh=(sm[0]+sm[1])&np.uint64(255)
                output=[]
                for p in (0,1):
                    g=keys[p]['horner_product']-opened*keys[p]['horner_rs']-sh*keys[p]['input_mask']+values[p][:,1]+keys[p]['refresh']
                    if p==0:g+=opened*sh
                    output.append(g)
                assert np.array_equal((output[0]+output[1])&np.uint64(255),f[(h-alpha.astype(np.int64))%256].view(np.uint64)&np.uint64(255))
        pairs+=n
    out=dict(mask_input_pairs_per_mode=pairs,modes=['selector_horner','weighted_vector4'],
        individual_party_kernel_checks=True,all_passed=True,native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Functional and independent query-path checks only. Underlying PRG serialization checked separately. No implementation security, optimality or novelty claim.')
    (ROOT/'results/public_spline_selector_check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)


if __name__=='__main__':main()
