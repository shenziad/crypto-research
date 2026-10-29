"""Typed DCF and modular private-contraction CUDA checks."""
import hashlib
import json
import numpy as np
from dcf_aes_native import generate_batch,evaluate_batch
from dcf_cuda import tensor,select,contract,PROTOCOL_LIB
from fused_gated_lut import ROOT


def main():
    rng=np.random.default_rng(20261029);records=[]
    for bits in (6,32,63):
        for dim in (1,2):
            n=128;alpha=rng.integers(0,1<<bits,n,dtype=np.uint64);beta=rng.integers(0,1<<32,(n,dim),dtype=np.uint64)
            keys=generate_batch(alpha,beta,bits,32);x=rng.integers(0,1<<bits,(n,5),dtype=np.uint64);x[:,0]=alpha;x[:,1]=(alpha-np.uint64(1))&np.uint64((1<<bits)-1)
            co=np.arange(dim,dtype=np.int32);outputs=[]
            for p in (0,1):
                result=select(tensor(keys[p]),tensor(x),tensor(co),p,bits,dim,32,True).cpu().numpy().view(np.uint64)
                ref=np.stack([evaluate_batch(keys[p],x[:,j],p,bits,dim,32) for j in range(5)],axis=1)
                assert np.array_equal(result,ref);outputs.append(result)
            expected=(x<alpha[:,None])[:,:,None].astype(np.uint64)*beta[:,None,:]
            assert np.array_equal((outputs[0]+outputs[1])&np.uint64((1<<32)-1),expected)
            records.append(dict(bits=bits,group_bits=32,dim=dim,individual_parties_equal=True))
    mask=np.uint64((1<<63)-1);matrix_cases=[]
    for n in (1,127,129,4864):
        m=7;d=rng.integers(0,1<<63,(m,n),dtype=np.uint64);a=rng.integers(0,1<<63,(m,n),dtype=np.uint64)
        v=rng.integers(0,1<<63,n,dtype=np.uint64);z=rng.integers(0,1<<63,n,dtype=np.uint64);t=rng.integers(0,1<<63,m,dtype=np.uint64)
        for p in (0,1):
            actual=contract(tensor(d),tensor(a),tensor(v),tensor(z),tensor(t),p).cpu().numpy().view(np.uint64)
            expected=np.asarray([(int(t[i])+sum(-int(d[i,j])*int(v[j])-int(a[i,j])*int(z[j])+(int(d[i,j])*int(z[j]) if p==0 else 0) for j in range(n)))%(1<<63) for i in range(m)],dtype=np.uint64)
            assert np.array_equal(actual,expected),(n,p)
        matrix_cases.append(dict(m=m,n=n,python_bigint_all_rows_equal=True))
    result=dict(all_passed=True,typed_dcf_cases=records,matrix_cases=matrix_cases,native_sha256=hashlib.sha256(PROTOCOL_LIB.read_bytes()).hexdigest(),
        limits='Individual-party functionality and full-width modular arithmetic checks; no standalone security audit or end-to-end inference claim.')
    (ROOT/'results/dcf_cuda_protocol_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':main()
