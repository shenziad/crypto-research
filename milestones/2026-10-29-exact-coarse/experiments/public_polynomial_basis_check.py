"""Small-domain native checks of degrees0..3, including mixed ring bodies."""
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from dcf_wide_native import LIB
from public_polynomial_basis import prepare,evaluate


def main():
    rng=np.random.default_rng(20261029);rows=[]
    for view,body in ((6,6),(6,16)):
        q=1<<view;r=1<<body;starts=[0,1,11,32,33,57,63]
        alpha=np.arange(q,dtype=np.uint64)
        if body>view:alpha+=rng.integers(0,r//q,q,dtype=np.uint64)*q
        for degree in (0,1,2,3):
            coefficients=rng.integers(-19,20,(len(starts),degree+1)).tolist()
            for weighted in (False,True):
                beta=rng.integers(0,r,q,dtype=np.uint64) if weighted else np.ones(q,dtype=np.uint64)
                keys=prepare(starts,coefficients,alpha,beta,view,body);pairs=0
                for xl in range(q):
                    x=xl if xl<q//2 else xl-q
                    # Same-ring mode uses modular polynomial evaluation; signed
                    # lifts are equivalent mod r. Wider bodies use promised x.
                    h=(alpha+np.uint64(x%r))&np.uint64(r-1)
                    shares=[evaluate(keys[p],h,starts,coefficients,p,view,body) for p in (0,1)]
                    piece=max(i for i,s in enumerate(starts) if s<=xl)
                    plain=sum(c*pow(x,k) for k,c in enumerate(coefficients[piece]))
                    target=np.asarray([(int(b)*plain)%r for b in beta],dtype=np.uint64)
                    assert np.array_equal((shares[0]+shares[1])&np.uint64(r-1),target)
                    pairs+=q
                rows.append(dict(view_bits=view,body_bits=body,degree=degree,weighted=weighted,pairs=pairs,passed=True))
                print(rows[-1],flush=True)
    out=dict(records=rows,all_passed=True,total_pairs=sum(v['pairs'] for v in rows),
        native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Generic public algebra/functional checks with standard vectorDCF. No optimized degree>1 native traversal, protocol composition proof, novelty or model accuracy claim.')
    (ROOT/'results/public_polynomial_basis_check.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
