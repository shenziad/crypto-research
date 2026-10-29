"""Variable-width vector DCF adapter for the matched long-payload baseline."""
import ctypes
import os
from pathlib import Path
import numpy as np

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_wide_native.so';_lib=None


def library():
    global _lib
    if _lib is None:_lib=ctypes.CDLL(str(LIB))
    return _lib


def size(bits,dim):return 16+bits*(17+8*dim)+8*dim


def generate(alpha,beta,bits,roots=None):
    alpha=np.ascontiguousarray(alpha,dtype=np.uint64);beta=np.ascontiguousarray(beta,dtype=np.uint64)
    n,dim=beta.shape;assert len(alpha)==n and 1<=dim<=128 and 1<=bits<=63
    roots=np.frombuffer(os.urandom(n*32),dtype=np.uint8).copy() if roots is None else np.ascontiguousarray(roots,dtype=np.uint8)
    assert roots.size==n*32
    keys=[np.empty((n,size(bits,dim)),dtype=np.uint8) for _ in (0,1)]
    fn=library().dcf_wide_generate;fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5;fn.restype=ctypes.c_int
    assert fn(bits,dim,n,alpha.ctypes.data,beta.ctypes.data,roots.ctypes.data,keys[0].ctypes.data,keys[1].ctypes.data)==0
    return keys


def select(key,x,coords,party,bits,dim):
    key=np.ascontiguousarray(key,dtype=np.uint8);x=np.ascontiguousarray(x,dtype=np.uint64)
    coords=np.ascontiguousarray(coords,dtype=np.int32);n=len(x)
    assert key.shape==(n,size(bits,dim))
    out=np.empty((n,len(coords)),dtype=np.uint64)
    fn=library().dcf_wide_select;fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*4;fn.restype=ctypes.c_int
    assert fn(bits,dim,n,party,len(coords),key.ctypes.data,x.ctypes.data,coords.ctypes.data,out.ctypes.data)==0
    return out


def spline(key,opened,starts,party,bits,cached=True):
    opened=np.ascontiguousarray(opened,dtype=np.uint64);starts=np.ascontiguousarray(starts,dtype=np.uint64)
    n=len(opened);m=len(starts);assert key.shape==(n,size(bits,2*m))
    out=np.empty(n,dtype=np.uint64)
    fn=library().dcf_wide_spline;fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*4;fn.restype=ctypes.c_int
    assert fn(bits,m,n,party,int(cached),key.ctypes.data,opened.ctypes.data,starts.ctypes.data,out.ctypes.data)==0
    return out


def spline_endpoint(key,opened,starts,slopes,intercepts,party,bits,basis):
    arrays=[np.ascontiguousarray(v,dtype=np.uint64) for v in (opened,starts,slopes,intercepts)]
    opened,starts,slopes,intercepts=arrays;n=len(opened);m=len(starts)
    assert len(slopes)==len(intercepts)==m and key.shape==(n,size(bits,2 if basis else 2*m))
    out=np.empty(n,dtype=np.uint64);fn=library().dcf_spline_endpoint
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*6;fn.restype=ctypes.c_int
    assert fn(bits,m,n,party,int(basis),key.ctypes.data,opened.ctypes.data,starts.ctypes.data,slopes.ctypes.data,intercepts.ctypes.data,out.ctypes.data)==0
    return out
