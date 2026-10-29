"""Exact aligned spline: one shared low-bit correction, independent coarse mask.

Standard deterministic truncate-reduce plus separated classifier/body FSS.
This is a stronger constructive baseline, not a new-primitive claim.
"""
import ctypes
import numpy as np
from dcf_aes_native import generate_batch,evaluate_batch
from dcf_wide_native import generate
from dithered_spline_fss import parameters,classifier_bits,LIB,MODES
from fused_gated_lut import random64,split64

_lib=None


def prepare(n,spec,bits=63,variant='basis',alpha=None,gamma=None):
    assert variant in ('basis','weighted') and not spec.get('full_classifier_reference',False)
    mask=np.uint64((1<<bits)-1);view=classifier_bits(spec);vm=np.uint64((1<<view)-1);shift=spec['classification_shift']
    alpha=random64((n,))&mask if alpha is None else np.asarray(alpha,dtype=np.uint64)
    gamma=random64((n,))&vm if gamma is None else np.asarray(gamma,dtype=np.uint64)
    assert alpha.shape==gamma.shape==(n,) and np.all(alpha<=mask) and np.all(gamma<=vm)
    starts,ends,slopes,intercepts=parameters(spec,bits)
    al=(gamma[:,None]+starts)&vm;ar=(gamma[:,None]+ends)&vm;ne=(ends+np.uint64(1))&vm;ar1=(gamma[:,None]+ne)&vm
    cr=((al>ar).astype(np.int64)-(al>starts).astype(np.int64)+(ar1>ne).astype(np.int64)+(ar==vm).astype(np.int64)).view(np.uint64)
    cshift=intercepts[None,:]-slopes[None,:]*alpha[:,None]
    e0=(cr*slopes[None,:]).sum(1,dtype=np.uint64);e1=(cr*cshift).sum(1,dtype=np.uint64)
    xi=np.stack([np.ones(n,dtype=np.uint64),alpha],1);corr=np.stack([e0,e1],1)
    rb=None
    if variant=='weighted':
        rb=random64((n,))&mask;xi=np.concatenate([xi,-rb[:,None]*xi],1);corr=np.concatenate([corr,-rb[:,None]*corr],1)
    dcf=generate((gamma-np.uint64(1))&vm,xi&mask,view)
    low=generate_batch(alpha&np.uint64((1<<shift)-1),np.full((n,1),(1<<32)-1,dtype=np.uint64),shift,32)
    high_alpha=(alpha&np.uint64((1<<spec['input_bits'])-1))>>np.uint64(shift)
    qconst=split64((gamma-high_alpha)&np.uint64((1<<32)-1),32)
    def share(v):return [p&mask for p in split64(v)]
    ins=share(alpha);bs=share(xi);es=share(corr);zs=share(np.zeros((n,2) if variant=='weighted' else (n,),dtype=np.uint64))
    keys=[dict(dcf=dcf[p],input_mask=ins[p],basis_share=bs[p],correction_share=es[p],refresh=zs[p],
        classifier_low=low[p],classifier_constant=qconst[p]) for p in (0,1)]
    if rb is not None:
        rbs=share(rb)
        for p in (0,1):keys[p]['b_mask']=rbs[p]
    return keys,alpha,gamma


def masked_classifier(key,opened,spec,party):
    h=np.asarray(opened,dtype=np.uint64);shift=spec['classification_shift'];view=classifier_bits(spec)
    low=evaluate_batch(key['classifier_low'],h&np.uint64((1<<shift)-1),party,shift,1,32)[:,0]
    result=key['classifier_constant'].astype(np.uint64)+low
    if party==0:result+=(h&np.uint64((1<<spec['input_bits'])-1))>>np.uint64(shift)
    return result&np.uint64((1<<view)-1)


def raw(key,h,classifier,spec,party,bits=63,variant='basis'):
    global _lib
    s,e,m,c=parameters(spec,bits);h=np.ascontiguousarray(h,dtype=np.uint64);hc=np.ascontiguousarray(classifier,dtype=np.uint64)
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));_lib.dcf_dither_spline.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*7;_lib.dcf_dither_spline.restype=ctypes.c_int
    out=np.empty((len(h),2),dtype=np.uint64)
    assert _lib.dcf_dither_spline(classifier_bits(spec),len(s),len(h),party,MODES[variant],key.ctypes.data,h.ctypes.data,hc.ctypes.data,s.ctypes.data,m.ctypes.data,c.ctypes.data,out.ctypes.data)==0
    return out


def evaluate(key,opened,classifier,spec,party,bits=63,variant='basis'):
    h=np.asarray(opened,dtype=np.uint64);hc=np.asarray(classifier,dtype=np.uint64);s,e,m,c=parameters(spec,bits)
    vm=np.uint64((1<<classifier_bits(spec))-1);mask=np.uint64((1<<bits)-1)
    cx=((hc[:,None]>s).astype(np.int64)-(hc[:,None]>((e+np.uint64(1))&vm)).astype(np.int64)).view(np.uint64)
    c0=(cx*(m[None,:]*h[:,None]+c[None,:])).sum(1,dtype=np.uint64);c1=(cx*(-m[None,:])).sum(1,dtype=np.uint64)
    out=raw(key['dcf'],h,hc,spec,party,bits,variant)
    if variant=='weighted':
        for j in (0,1):out[:,j]+=c0*key['basis_share'][:,2*j]+c1*key['basis_share'][:,2*j+1]+h*key['correction_share'][:,2*j]+key['correction_share'][:,2*j+1]+key['refresh'][:,j]
        return out&mask
    return (out[:,0]+c0*key['basis_share'][:,0]+c1*key['basis_share'][:,1]+h*key['correction_share'][:,0]+key['correction_share'][:,1]+key['refresh'])&mask
