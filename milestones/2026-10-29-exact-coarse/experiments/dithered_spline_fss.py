"""Public-dither separated classifier/body specialization of standard FSS.

No independent novelty claim. Same four payload layouts as matched references;
the classification function is randomized, body input retains full precision.
"""
import ctypes
from pathlib import Path
import numpy as np
from dcf_wide_native import generate
from fused_gated_lut import random64,split64
from public_spline_coefficient_basis import parameters as fine_parameters

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_dither_native.so';_lib=None
MODES={'selector':0,'basis':1,'weighted':2,'wide':3}


def classifier_bits(spec):
    return spec['input_bits'] if spec.get('full_classifier_reference',False) else spec['input_bits']-spec['classification_shift']


def parameters(spec,bits=63):
    s,e,m,c=fine_parameters(spec,bits);shift=spec['classification_shift'];bucket=np.uint64(1<<shift)
    assert np.all(s%bucket==0) and np.all((e+np.uint64(1))%bucket==0)
    return (s,e,m,c) if spec.get('full_classifier_reference',False) else (s>>np.uint64(shift),e>>np.uint64(shift),m,c)


def public_classifier(opened,spec):
    h=np.asarray(opened,dtype=np.uint64);shift=spec['classification_shift'];guard=np.uint64((1<<spec['input_bits'])-1)
    if spec.get('full_classifier_reference',False):
        rho=h&np.uint64((1<<shift)-1)
        return (h+np.uint64((1<<shift)-1)-rho)&guard
    return (h&guard)>>np.uint64(shift)


def prepare(n,spec,bits=63,alpha=None,variant='basis'):
    assert variant in MODES;mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits)
    view=classifier_bits(spec);vm=np.uint64((1<<view)-1)
    alpha=random64((n,))&mask if alpha is None else np.asarray(alpha,dtype=np.uint64)
    assert alpha.shape==(n,) and np.all(alpha<=mask)
    ca=(alpha&vm) if spec.get('full_classifier_reference',False) else public_classifier(alpha,spec)
    gamma=(ca-np.uint64(1))&vm;al=(ca[:,None]+s[None,:])&vm;ar=(ca[:,None]+e[None,:])&vm
    ne=(e+np.uint64(1))&vm;ar1=(ca[:,None]+ne[None,:])&vm
    cr=((al>ar).astype(np.int64)-(al>s[None,:]).astype(np.int64)+(ar1>ne[None,:]).astype(np.int64)+(ar==vm).astype(np.int64)).view(np.uint64)
    def share(v):return [x&mask for x in split64(v)]
    inputs=share(alpha);xi=np.stack([np.ones(n,dtype=np.uint64),alpha],axis=1);cshift=c[None,:]-m[None,:]*alpha[:,None]
    if variant=='selector':
        payload=np.ones((n,1),dtype=np.uint64);correction=np.stack([(cr*m[None,:]).sum(1,dtype=np.uint64),(cr*c[None,:]).sum(1,dtype=np.uint64)],1)
        dcf=generate(gamma,payload,view);es=share(correction);rs=random64((n,))&mask;rss=share(rs);product=share(alpha*rs);zero=share(np.zeros(n,dtype=np.uint64))
        return [dict(dcf=dcf[p],input_mask=inputs[p],correction_share=es[p],horner_rs=rss[p],horner_product=product[p],refresh=zero[p]) for p in (0,1)],alpha
    e0=(cr*m[None,:]).sum(1,dtype=np.uint64);e1=(cr*cshift).sum(1,dtype=np.uint64)
    if variant=='weighted':
        rb=random64((n,))&mask;xi=np.stack([np.ones(n,dtype=np.uint64),alpha,-rb,-rb*alpha],1)&mask
        correction=np.stack([e0,e1,-rb*e0,-rb*e1],1);rbs=share(rb);zeros=share(np.zeros((n,2),dtype=np.uint64));payload=xi
    else:
        correction=np.stack([e0,e1],1);zeros=share(np.zeros(n,dtype=np.uint64))
        payload=np.stack([np.broadcast_to(m,cshift.shape),cshift],2).reshape(n,-1)&mask if variant=='wide' else xi
    dcf=generate(gamma,payload,view);bs=share(xi);es=share(correction)
    keys=[dict(dcf=dcf[p],input_mask=inputs[p],basis_share=bs[p],correction_share=es[p],refresh=zeros[p]) for p in (0,1)]
    if variant=='weighted':
        for p in (0,1):keys[p]['b_mask']=rbs[p]
    return keys,alpha


def raw(key,opened,spec,party,bits=63,variant='basis'):
    global _lib
    s,e,m,c=parameters(spec,bits);h=np.ascontiguousarray(opened,dtype=np.uint64);hc=np.ascontiguousarray(public_classifier(h,spec));view=classifier_bits(spec)
    dim=1 if variant=='selector' else 2 if variant=='basis' else 4 if variant=='weighted' else 2*len(s)
    assert key.shape==(len(h),16+view*(17+8*dim)+8*dim)
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));_lib.dcf_dither_spline.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*7;_lib.dcf_dither_spline.restype=ctypes.c_int
    out=np.empty((len(h),2),dtype=np.uint64)
    assert _lib.dcf_dither_spline(view,len(s),len(h),party,MODES[variant],key.ctypes.data,h.ctypes.data,hc.ctypes.data,s.ctypes.data,m.ctypes.data,c.ctypes.data,out.ctypes.data)==0
    return out


def evaluate(key,opened,spec,party,bits=63,variant='basis'):
    mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits);h=np.asarray(opened,dtype=np.uint64);hc=public_classifier(h,spec)
    vm=np.uint64((1<<classifier_bits(spec))-1)
    cx=((hc[:,None]>s[None,:]).astype(np.int64)-(hc[:,None]>((e+np.uint64(1))&vm)[None,:]).astype(np.int64)).view(np.uint64)
    out=raw(key['dcf'],h,spec,party,bits,variant)
    if variant=='selector':
        if party==0:out+=np.stack([(cx*m[None,:]).sum(1,dtype=np.uint64),(cx*c[None,:]).sum(1,dtype=np.uint64)],1)
        return (out+key['correction_share'])&mask
    c0=(cx*(m[None,:]*h[:,None]+c[None,:])).sum(1,dtype=np.uint64);c1=(cx*(-m[None,:])).sum(1,dtype=np.uint64)
    if variant=='weighted':
        for j in (0,1):out[:,j]+=c0*key['basis_share'][:,2*j]+c1*key['basis_share'][:,2*j+1]+h*key['correction_share'][:,2*j]+key['correction_share'][:,2*j+1]+key['refresh'][:,j]
        return out&mask
    return (out[:,0]+c0*key['basis_share'][:,0]+c1*key['basis_share'][:,1]+h*key['correction_share'][:,0]+key['correction_share'][:,1]+key['refresh'])&mask
