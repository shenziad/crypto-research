"""Public coefficient selection+Horner reference and vector4 gated basis."""
import ctypes
from pathlib import Path
import numpy as np
from fused_gated_lut import random64,split64
from dcf_aes_native import generate_batch
from dcf_wide_native import generate as wide_generate
from public_spline_coefficient_basis import parameters

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_selector_native.so';_lib=None


def prepare(n,spec,bits=63,alpha=None,weighted=False):
    mask=np.uint64((1<<bits)-1);view=spec.get('input_bits',bits);vm=np.uint64((1<<view)-1)
    s,e,m,c=parameters(spec,bits);alpha=random64((n,))&mask if alpha is None else np.asarray(alpha,dtype=np.uint64)
    gamma=(alpha-np.uint64(1))&vm
    al=(alpha[:,None]+s[None,:])&vm;ar=(alpha[:,None]+e[None,:])&vm;ne=(e+np.uint64(1))&vm
    ar1=(alpha[:,None]+ne[None,:])&vm
    cr=((al>ar).astype(np.int64)-(al>s[None,:]).astype(np.int64)+(ar1>ne[None,:]).astype(np.int64)+(ar==vm).astype(np.int64)).view(np.uint64)
    def share(x):return [v&mask for v in split64(x)]
    inputs=share(alpha)
    if weighted:
        rb=random64((n,))&mask;basis=np.stack([np.ones(n,dtype=np.uint64),alpha,-rb,-rb*alpha],axis=1)&mask
        dcf=wide_generate(gamma,basis,view)
        cshift=(c[None,:]-m[None,:]*alpha[:,None])&mask
        e0=(cr*m[None,:]).sum(axis=1,dtype=np.uint64);e1=(cr*cshift).sum(axis=1,dtype=np.uint64)
        correction=np.stack([e0,e1,-rb*e0,-rb*e1],axis=1)&mask
        bs=share(basis);es=share(correction);refresh=share(np.zeros((n,2),dtype=np.uint64));rbs=share(rb)
        return [dict(dcf=dcf[p],input_mask=inputs[p],basis_share=bs[p],correction_share=es[p],refresh=refresh[p],b_mask=rbs[p]) for p in (0,1)],alpha
    dcf=generate_batch(gamma,np.ones((n,1),dtype=np.uint64),view,64)
    correction=np.stack([(cr*m[None,:]).sum(axis=1,dtype=np.uint64),(cr*c[None,:]).sum(axis=1,dtype=np.uint64)],axis=1)&mask
    es=share(correction);refresh=share(np.zeros(n,dtype=np.uint64))
    rs=random64((n,))&mask;rss=share(rs);product=share(alpha*rs)
    return [dict(dcf=dcf[p],input_mask=inputs[p],correction_share=es[p],refresh=refresh[p],horner_rs=rss[p],horner_product=product[p]) for p in (0,1)],alpha


def native(key,opened,spec,party,bits=63,weighted=False):
    global _lib
    starts,ends,slopes,intercepts=parameters(spec,bits);view=spec.get('input_bits',bits)
    opened=np.ascontiguousarray(opened,dtype=np.uint64);n=len(opened);dim=4 if weighted else 1
    assert key.shape==(n,16+view*(17+8*dim)+8*dim)
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));_lib.dcf_public_coefficient.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*6
        _lib.dcf_public_coefficient.restype=ctypes.c_int
    output=np.empty((n,2),dtype=np.uint64)
    assert _lib.dcf_public_coefficient(view,len(starts),n,party,int(weighted),key.ctypes.data,opened.ctypes.data,
        starts.ctypes.data,slopes.ctypes.data,intercepts.ctypes.data,output.ctypes.data)==0
    return output


def evaluate(key,opened,spec,party,bits=63,weighted=False):
    mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits);h=np.asarray(opened,dtype=np.uint64)
    vm=np.uint64((1<<spec.get('input_bits',bits))-1);low=h&vm
    cx=((low[:,None]>s[None,:]).astype(np.int64)-(low[:,None]>((e+np.uint64(1))&vm)[None,:]).astype(np.int64)).view(np.uint64)
    out=native(key['dcf'],h,spec,party,bits,weighted)
    if weighted:
        c0=(cx*(m[None,:]*h[:,None]+c[None,:])).sum(axis=1,dtype=np.uint64)
        c1=(cx*(-m[None,:])).sum(axis=1,dtype=np.uint64)
        for j in (0,1):
            out[:,j]+=c0*key['basis_share'][:,2*j]+c1*key['basis_share'][:,2*j+1]
            out[:,j]+=h*key['correction_share'][:,2*j]+key['correction_share'][:,2*j+1]+key['refresh'][:,j]
    else:
        if party==0:
            out[:,0]+=(cx*m[None,:]).sum(axis=1,dtype=np.uint64)
            out[:,1]+=(cx*c[None,:]).sum(axis=1,dtype=np.uint64)
        out+=key['correction_share']
    return out&mask
