"""Scalar public affine function from a DCF payload basis(1,alpha).

Constructive reference in mixed input/output rings. Standard DCF, public linear
postprocessing and integer interval aggregation; independent novelty unconfirmed.
"""
import ctypes
from pathlib import Path
import numpy as np
from dcf_aes_native import generate_batch,evaluate_batch
from fused_gated_lut import random64,split64

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_spline_native.so';_lib=None


def prepare(n,bits=32,alpha=None):
    mask=np.uint64((1<<bits)-1)
    alpha=random64((n,))&mask if alpha is None else np.ascontiguousarray(alpha,dtype=np.uint64)
    assert len(alpha)==n and np.all(alpha<=mask)
    keys=generate_batch(alpha,np.stack([np.ones(n,dtype=np.uint64),alpha],axis=1),bits,64)
    inputs=split64(alpha,32 if bits<=32 else 64);constants=split64(alpha)
    zero=split64(np.zeros(n,dtype=np.uint64))
    return [dict(dcf=keys[p],input_mask=inputs[p],alpha_share=constants[p],refresh=zero[p]) for p in (0,1)],alpha


def queries(spec,opened,bits=32):
    h=np.ascontiguousarray(opened,dtype=np.uint64);hs=h.astype(np.int64);mask=np.uint64((1<<bits)-1)
    runs=spec['delta_runs'];total=sum(s*(b-a+1) for a,b,s in runs)
    constant=np.full(len(h),spec['function_at_last'],dtype=np.int64);alpha_coefficient=np.zeros(len(h),dtype=np.int64)
    points=[];weights=[]
    def add(q,w0,w1):
        points.append(q);weights.append(np.stack([np.broadcast_to(np.asarray(w0,dtype=np.int64),h.shape),np.broadcast_to(np.asarray(w1,dtype=np.int64),h.shape)],axis=1).view(np.uint64))
    if total:add(h,total,0)
    for a,b,s in runs:
        constant-=s*np.maximum(0,b-np.maximum(a,hs+1)+1)
        if a==b:add((h-np.uint64(a))&mask,-s,0);continue
        left=(h-np.uint64(b))&mask;right=(h-np.uint64(a))&mask;previous=(left-np.uint64(1))&mask
        add(right,-s*(right.astype(np.int64)+1),s)
        add(previous,s*(previous.astype(np.int64)+1),-s)
        alpha_coefficient-=s*((left==0)|(left>right)).astype(np.int64)
    if not points:add(h,0,0)
    return np.ascontiguousarray(np.stack(points,axis=1)),np.ascontiguousarray(np.stack(weights,axis=1)),constant.view(np.uint64),alpha_coefficient.view(np.uint64)


def linear(key,q,w,party,bits=32,cached=True):
    global _lib
    n,count=q.shape;assert key.shape==(n,16+bits*33+16) and w.shape==(n,count,2)
    assert key.dtype==np.uint8 and q.dtype==w.dtype==np.uint64 and all(v.flags.c_contiguous for v in (key,q,w))
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));_lib.dcf_linear_queries.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*4
        _lib.dcf_linear_queries.restype=ctypes.c_int
    out=np.empty(n,dtype=np.uint64)
    assert _lib.dcf_linear_queries(bits,n,party,count,int(cached),key.ctypes.data,q.ctypes.data,w.ctypes.data,out.ctypes.data)==0
    return out


def evaluate(key,opened,spec,party,bits=32,cached=True,compiled=None):
    q,w,constant,alpha_coefficient=queries(spec,opened,bits) if compiled is None else compiled
    out=linear(key['dcf'],q,w,party,bits,cached)+alpha_coefficient*key['alpha_share']+key['refresh']
    if party==0:out+=constant
    return out
