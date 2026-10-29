"""Public affine coefficient-basis specialization of the2021 spline layout.

Same-ring63 backend encoded in group64. One vector2 DCF carries (1,alpha),
and public matrices reconstruct each interval's shifted coefficients. Secret
interval corrections are aggregated into two shared coefficients. Not a claim
of an independently new primitive or complete two-party protocol proof.
"""
import numpy as np
from dcf_aes_native import generate_batch
from fused_gated_lut import random64,split64
from public_spline_fss import linear


def parameters(spec,bits):
    mask=(1<<bits)-1;view=spec.get('input_bits',bits);view_ring=1<<view
    pieces=[]
    for a,b,m,c in spec['unsigned_pieces']:
        for lo,hi in ((a,min(b,view_ring//2)),(max(a,view_ring//2),b)):
            if lo<hi:pieces.append((lo,hi,m,c))
    starts=np.asarray([v[0] for v in pieces],dtype=np.uint64)
    ends=np.asarray([v[1]-1 for v in pieces],dtype=np.uint64)
    slopes=np.asarray([v[2]% (1<<bits) for v in pieces],dtype=np.uint64)
    intercepts=np.asarray([(c+(m*view_ring if a>=view_ring//2 else 0))%(1<<bits) for a,b,m,c in pieces],dtype=np.uint64)
    assert starts[0]==0 and int(ends[-1])==view_ring-1
    return starts,ends,slopes,intercepts


def prepare(n,spec,bits=63,alpha=None,variant='basis'):
    assert variant in ('basis','wide')
    mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits)
    view=spec.get('input_bits',bits);vmask=np.uint64((1<<view)-1)
    alpha=random64((n,))&mask if alpha is None else np.ascontiguousarray(alpha,dtype=np.uint64)
    assert len(alpha)==n and np.all(alpha<=mask)
    basis=np.stack([np.ones(n,dtype=np.uint64),alpha],axis=1)
    gamma=(alpha-np.uint64(1))&vmask
    al=(alpha[:,None]+s[None,:])&vmask;ar=(alpha[:,None]+e[None,:])&vmask
    next_e=(e+np.uint64(1))&vmask;ar1=(alpha[:,None]+next_e[None,:])&vmask
    correction=(al>ar).astype(np.int64)-(al>s[None,:]).astype(np.int64)+(ar1>next_e[None,:]).astype(np.int64)+(ar==vmask).astype(np.int64)
    shifted_c=c[None,:]-m[None,:]*alpha[:,None]
    if variant=='basis':dcf=generate_batch(gamma,basis,view,64)
    else:
        from dcf_wide_native import generate
        payload=np.stack([np.broadcast_to(m,shifted_c.shape),shifted_c],axis=2).reshape(n,-1)&mask
        dcf=generate(gamma,payload,view)
    E=np.stack([(correction.view(np.uint64)*m[None,:]).sum(axis=1,dtype=np.uint64),
        (correction.view(np.uint64)*shifted_c).sum(axis=1,dtype=np.uint64)],axis=1)&mask
    def shares(v):return [a&mask for a in split64(v)]
    bs=shares(basis);es=shares(E);inputs=shares(alpha);refresh=shares(np.zeros(n,dtype=np.uint64))
    return [dict(dcf=dcf[p],basis_share=bs[p],correction_share=es[p],input_mask=inputs[p],refresh=refresh[p]) for p in (0,1)],alpha


def compile_constants(spec,opened,bits=63):
    mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits);h=np.asarray(opened,dtype=np.uint64)
    vm=np.uint64((1<<spec.get('input_bits',bits))-1);low=h&vm
    cx=(low[:,None]>s[None,:]).astype(np.int64)-(low[:,None]>((e+np.uint64(1))&vm)[None,:]).astype(np.int64)
    constant0=(cx.view(np.uint64)*(m[None,:]*h[:,None]+c[None,:])).sum(axis=1,dtype=np.uint64)
    constant1=(cx.view(np.uint64)*(-m[None,:])).sum(axis=1,dtype=np.uint64)
    return constant0,constant1


def compile_queries(spec,opened,bits=63):
    mask=np.uint64((1<<bits)-1);s,e,m,c=parameters(spec,bits)
    vm=np.uint64((1<<spec.get('input_bits',bits))-1)
    h=np.asarray(opened,dtype=np.uint64);q=(h[:,None]-s[None,:]-np.uint64(1))&vm
    dm=np.roll(m,1)-m;dc=np.roll(c,1)-c
    weights=np.stack([dm[None,:]*h[:,None]+dc[None,:],np.broadcast_to(-dm,q.shape)],axis=2)&mask
    constant0,constant1=compile_constants(spec,opened,bits)
    return np.ascontiguousarray(q),np.ascontiguousarray(weights),constant0,constant1


def evaluate(key,opened,spec,party,bits=63,compiled=None,cached=True,variant='basis'):
    view=spec.get('input_bits',bits)
    if cached==3:
        from dcf_wide_native import spline_endpoint
        c0,c1=compile_constants(spec,opened,bits) if compiled is None else compiled
        starts,_,slopes,intercepts=parameters(spec,bits)
        out=spline_endpoint(key['dcf'],opened,starts,slopes,intercepts,party,view,variant=='basis')
    elif variant=='basis':
        q,w,c0,c1=compile_queries(spec,opened,bits) if compiled is None else compiled
        out=linear(key['dcf'],q,w,party,view,cached)
    else:
        from dcf_wide_native import spline
        c0,c1=compile_constants(spec,opened,bits) if compiled is None else compiled
        out=spline(key['dcf'],opened,parameters(spec,bits)[0],party,view,cached)
    out+=c0*key['basis_share'][:,0]+c1*key['basis_share'][:,1]
    out+=opened*key['correction_share'][:,0]+key['correction_share'][:,1]+key['refresh']
    return out&np.uint64((1<<bits)-1)
