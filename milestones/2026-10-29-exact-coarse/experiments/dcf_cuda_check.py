"""Independent AES and individual-party CPU/CUDA DCF share equality."""
import hashlib
import json
import numpy as np
import torch
from dcf_wide_native import generate,select as cpu_select
from dcf_cuda import tensor,aes,select,LIB
from fused_gated_lut import ROOT


def main():
    rng=np.random.default_rng(20261029);n=1024
    path=ROOT/'results/dcf_cuda_aes_reference.npz';manifest=json.loads((ROOT/'results/dcf_cuda_aes_reference.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    with np.load(path) as ref:keys,plain,expected=[ref[k] for k in ('keys','plain','expected')]
    actual=aes(tensor(keys),tensor(plain)).cpu().numpy()
    assert actual[0].tobytes().hex()=='69c4e0d86a7b0430d8cdb78070b4c55a'
    assert np.array_equal(actual,expected);print('AES independent1024 cases passed',flush=True)
    records=[]
    for bits in (6,32,63):
        for dim in (1,2,4,52):
            n=128;alpha=rng.integers(0,1<<bits,n,dtype=np.uint64);beta=rng.integers(0,1<<64,(n,dim),dtype=np.uint64)
            native=generate(alpha,beta,bits);x=rng.integers(0,1<<bits,(n,5),dtype=np.uint64)
            x[:,0]=alpha;x[:,1]=(alpha-np.uint64(1))&np.uint64((1<<bits)-1);x[:,2]=0;x[:,3]=(1<<bits)-1
            coords=np.asarray(sorted(set([0,dim//2,dim-1])),dtype=np.int32)
            outputs=[]
            for p in (0,1):
                result=select(tensor(native[p]),tensor(x),tensor(coords),p,bits,dim).cpu().numpy().view(np.uint64)
                reference=np.stack([cpu_select(native[p],x[:,j],coords,p,bits,dim) for j in range(5)],axis=1)
                assert np.array_equal(result,reference),(bits,dim,p)
                # Query-specific coordinate layout must agree with the same
                # global coordinates, including nonadjacent scalar words.
                tiled=np.tile(coords,(n,5,1));other=select(tensor(native[p]),tensor(x),tensor(tiled),p,bits,dim).cpu().numpy().view(np.uint64)
                assert np.array_equal(other,reference);outputs.append(result)
            target=(x<alpha[:,None])[:,:,None].astype(np.uint64)*beta[:,None,coords]
            assert np.array_equal(outputs[0]+outputs[1],target)
            records.append(dict(bits=bits,dim=dim,keys=n,queries_per_key=5,global_and_query_coordinates=True,individual_parties_equal=True))
            print(records[-1],flush=True)
    out=dict(aes_independent_cases=1024,records=records,all_passed=True,hardware=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256((ROOT/'experiments/dcf_cuda.cu').read_bytes()).hexdigest(),native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Functional AES/DCF share equality only; no security audit or original Sigma/FuseFSS GPU implementation comparison. Table-based device AES is a research backend.')
    (ROOT/'results/dcf_cuda_check.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
