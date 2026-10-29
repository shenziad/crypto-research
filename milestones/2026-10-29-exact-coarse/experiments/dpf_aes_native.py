"""ctypes adapter for the local AES-instruction DPF research backend."""
import ctypes
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'toolchain/dpf_aes_native.so'
_lib=None


def library():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB))
        _lib.dpf_generate_batch.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5+[ctypes.c_int]
        _lib.dpf_generate_batch.restype=ctypes.c_int
        _lib.dpf_lookup_batch.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*4+[ctypes.c_int]*2
        _lib.dpf_lookup_batch.restype=ctypes.c_int
    return _lib


def generate_batch(alpha,beta,roots=None,group_bits=64):
    alpha=np.ascontiguousarray(alpha,dtype=np.uint8)
    beta=np.ascontiguousarray(beta,dtype=np.uint64)
    n=len(alpha);assert beta.ndim==2 and len(beta)==n and beta.shape[1] in (1,2)
    assert group_bits in (16,32,64)
    dim=beta.shape[1];size=16+17*8+(group_bits//8)*dim
    roots=np.frombuffer(os.urandom(n*32),dtype=np.uint8).copy() if roots is None else np.ascontiguousarray(roots,dtype=np.uint8)
    assert roots.size==n*32
    keys=[np.empty((n,size),dtype=np.uint8) for _ in (0,1)]
    status=library().dpf_generate_batch(8,dim,n,alpha.ctypes.data,beta.ctypes.data,roots.ctypes.data,keys[0].ctypes.data,keys[1].ctypes.data,group_bits)
    assert status==0
    return keys


def lookup_batch(key,ahat,table,party,dim,group_bits=64):
    ahat=np.ascontiguousarray(ahat,dtype=np.uint8);key=np.ascontiguousarray(key,dtype=np.uint8)
    table=np.ascontiguousarray(table,dtype=np.uint64);n=len(ahat)
    assert group_bits in (16,32,64)
    assert key.shape==(n,16+17*8+(group_bits//8)*dim) and table.shape in ((256,),(n,256))
    out=np.empty((n,dim),dtype=np.uint64)
    status=library().dpf_lookup_batch(8,dim,n,party,key.ctypes.data,ahat.ctypes.data,table.ctypes.data,out.ctypes.data,0 if table.ndim==1 else 256,group_bits)
    assert status==0
    return out
