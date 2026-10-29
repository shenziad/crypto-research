"""Public polynomial mask-power basis reference, over rings without division.

Standard DCF and public linear postprocessing. This reference checks a
generalization of the implemented affine kernel, without a novelty claim.
Coefficients are polynomials in the signed/body ring variable, not unsigned
classifier coordinates. Each piece must stay on one side of the sign boundary.
"""
import math
import numpy as np
from dcf_wide_native import generate,select
from fused_gated_lut import random64,split64


def public_weights(coefficients,h):
    d=len(coefficients)-1
    return [sum(int(coefficients[k])*math.comb(k,l)*pow(int(h),k-l)*((-1)**l) for k in range(l,d+1)) for l in range(d+1)]


def prepare(starts,coefficients,alpha,beta,view_bits,body_bits,classifier_alpha=None):
    q=1<<view_bits;r=1<<body_bits;mask=np.uint64(r-1)
    starts=list(map(int,starts));n=len(alpha);dim=len(coefficients[0]);assert len(starts)==len(coefficients)
    assert starts[0]==0 and starts==sorted(set(starts)) and view_bits<=body_bits<=63
    assert all(len(c)==dim for c in coefficients)
    alpha=np.asarray(alpha,dtype=np.uint64);beta=np.asarray(beta,dtype=np.uint64)
    classifier_alpha=(alpha&np.uint64(q-1)) if classifier_alpha is None else np.asarray(classifier_alpha,dtype=np.uint64)
    assert classifier_alpha.shape==alpha.shape and np.all(classifier_alpha<q)
    xi=np.asarray([[int(b)*pow(int(a),l,r)%r for l in range(dim)] for a,b in zip(alpha,beta)],dtype=np.uint64)
    correction=np.zeros((n,dim),dtype=np.uint64)
    for row,(aa,bb) in enumerate(zip(alpha,beta)):
        aa=int(aa);bb=int(bb);ca=int(classifier_alpha[row])
        for i,s in enumerate(starts):
            e=(starts[i+1] if i+1<len(starts) else q)-1;ne=(e+1)%q
            al=(ca+s)%q;ar=(ca+e)%q;ar1=(ca+ne)%q
            cr=int(al>ar)-int(al>s)+int(ar1>ne)+int(ar==q-1)
            c=coefficients[i]
            for j in range(dim):
                v=sum(int(c[k])*math.comb(k,j)*pow(-aa,k-j) for k in range(j,dim))
                correction[row,j]=np.uint64((int(correction[row,j])+cr*bb*v)%r)
    keys=generate((classifier_alpha-np.uint64(1))&np.uint64(q-1),xi,view_bits)
    xs=split64(xi);es=split64(correction)
    return [dict(dcf=keys[p],basis=xs[p]&mask,correction=es[p]&mask) for p in (0,1)]


def evaluate(key,h,starts,coefficients,party,view_bits,body_bits,classifier_opened=None):
    q=1<<view_bits;r=1<<body_bits;dim=len(coefficients[0]);h=np.asarray(h,dtype=np.uint64)
    low=(h if classifier_opened is None else np.asarray(classifier_opened,dtype=np.uint64))&np.uint64(q-1)
    out=np.zeros(len(h),dtype=np.uint64);constant=np.zeros((len(h),dim),dtype=np.uint64)
    all_weights=np.asarray([[[(v%r) for v in public_weights(c,int(x))] for c in coefficients] for x in h],dtype=np.uint64)
    for i,s in enumerate(starts):
        ne=(starts[i+1] if i+1<len(starts) else q)%q
        cx=((low>np.uint64(s)).astype(np.int64)-(low>np.uint64(ne)).astype(np.int64)).view(np.uint64)
        constant+=cx[:,None]*all_weights[:,i,:]
        query=(low-np.uint64(s)-np.uint64(1))&np.uint64(q-1)
        shares=select(key['dcf'],query,np.arange(dim),party,view_bits,dim)
        difference=all_weights[:,(i-1)%len(starts),:]-all_weights[:,i,:]
        out+=(difference*shares).sum(axis=1,dtype=np.uint64)
    out+=(constant*key['basis']).sum(axis=1,dtype=np.uint64)
    powers=np.asarray([[pow(int(x),j,r) for j in range(dim)] for x in h],dtype=np.uint64)
    out+=(powers*key['correction']).sum(axis=1,dtype=np.uint64)
    return out&np.uint64(r-1)
