"""GPU resident public affine FSS, same serialized keys as CPU references.

Each public interval query has its own CUDA thread. Prefix sharing is absent
in this first GPU reference, unlike the optimized CPU endpoint implementation.
"""
import numpy as np
import torch
from dcf_cuda import tensor,select
from public_spline_coefficient_basis import parameters

MASK=(1<<63)-1


def prepare_runtime(key,opened,spec,variant):
    dither=spec.get('use_dither_fss',False)
    if dither:
        from dithered_spline_fss import parameters as dither_parameters,classifier_bits
        starts,ends,slopes,intercepts=dither_parameters(spec,63);view=classifier_bits(spec)
    else:starts,ends,slopes,intercepts=parameters(spec,63);view=spec['input_bits']
    n=len(opened);m=len(starts)
    h=tensor(opened);s,e,a,c=[tensor(v) for v in (starts,ends,slopes,intercepts)]
    query=((h[:,None]-s[None,:]-1)&((1<<view)-1)).contiguous()
    prev=(np.arange(m)-1)%m
    if variant=='wide':
        coords=np.stack([2*prev,2*prev+1,2*np.arange(m),2*np.arange(m)+1],axis=1).astype(np.int32)
        coords=tensor(np.tile(coords,(n,1,1)))
    else:coords=tensor(np.arange(4 if variant=='weighted' else 1 if variant=='selector' else 2,dtype=np.int32))
    runtime=dict(key={k:tensor(v) for k,v in key.items()},h=h,query=query,coords=coords,view=view,
        dim=2*m if variant=='wide' else 4 if variant=='weighted' else 1 if variant=='selector' else 2,
        variant=variant,starts=s,ends=e,slopes=a,intercepts=c,dm=torch.roll(a,1)-a,dc=torch.roll(c,1)-c,
        use_dither=dither,classification_shift=spec.get('classification_shift',0),guard_bits=spec['input_bits'],full_classifier_reference=spec.get('full_classifier_reference',False))
    return compile_public(runtime,h)


def compile_public(t,h):
    s,e,a,c=[t[k] for k in ('starts','ends','slopes','intercepts')];vm=(1<<t['view'])-1
    if t['use_dither']:
        shift=t['classification_shift']
        if t['full_classifier_reference']:low=(h+((1<<shift)-1)-(h&((1<<shift)-1)))&vm
        else:low=(h&((1<<t['guard_bits'])-1))>>shift
    else:low=h&vm
    cx=(low[:,None]>s[None,:]).to(torch.int64)-(low[:,None]>((e+1)&vm)[None,:]).to(torch.int64)
    t.update(h=h,query=((low[:,None]-s[None,:]-1)&vm).contiguous(),
        c0=(cx*(a[None,:]*h[:,None]+c[None,:])).sum(1),c1=(cx*(-a[None,:])).sum(1),
        selector_c0=(cx*a[None,:]).sum(1),selector_c1=(cx*c[None,:]).sum(1))
    return t


def raw(runtime,party):
    t=runtime;d=select(t['key']['dcf'],t['query'],t['coords'],party,t['view'],t['dim']);variant=t['variant']
    if variant=='wide':return (t['h'][:,None]*(d[:,:,0]-d[:,:,2])+d[:,:,1]-d[:,:,3]).sum(1)
    dm=t['dm'][None,:];dc=t['dc'][None,:]
    if variant=='selector':return torch.stack([(dm*d[:,:,0]).sum(1),(dc*d[:,:,0]).sum(1)],1)
    w0=dm*t['h'][:,None]+dc;w1=-dm
    if variant=='basis':return (w0*d[:,:,0]+w1*d[:,:,1]).sum(1)
    return torch.stack([(w0*d[:,:,2*j]+w1*d[:,:,2*j+1]).sum(1) for j in (0,1)],1)


def evaluate(runtime,party):
    t=runtime;k=t['key'];out=raw(t,party)
    if t['variant']=='selector':
        if party==0:out=out+torch.stack([t['selector_c0'],t['selector_c1']],1)
        return (out+k['correction_share'])&MASK
    if t['variant']=='weighted':
        correction=torch.stack([t['c0']*k['basis_share'][:,2*j]+t['c1']*k['basis_share'][:,2*j+1]
            +t['h']*k['correction_share'][:,2*j]+k['correction_share'][:,2*j+1]+k['refresh'][:,j] for j in (0,1)],1)
    else:correction=t['c0']*k['basis_share'][:,0]+t['c1']*k['basis_share'][:,1]+t['h']*k['correction_share'][:,0]+k['correction_share'][:,1]+k['refresh']
    return (out+correction)&MASK
