"""Adapter for the standard DCF correction-value algorithm, native AES backend."""
import ctypes
import os
import numpy as np
from dpf_aes_native import library


def key_size(bits,dim,group_bits):
    return 16+bits*(17+(group_bits//8)*dim)+(group_bits//8)*dim


def generate_batch(alpha,beta,bits,group_bits=64,roots=None):
    alpha=np.ascontiguousarray(alpha,dtype=np.uint64)
    beta=np.ascontiguousarray(beta,dtype=np.uint64)
    n=len(alpha);assert beta.ndim==2 and len(beta)==n and beta.shape[1] in (1,2)
    assert group_bits in (16,32,64) and 1<=bits<=63 and np.all(alpha<(1<<bits))
    dim=beta.shape[1]
    roots=np.frombuffer(os.urandom(n*32),dtype=np.uint8).copy() if roots is None else np.ascontiguousarray(roots,dtype=np.uint8)
    assert roots.size==32*n
    keys=[np.empty((n,key_size(bits,dim,group_bits)),dtype=np.uint8) for _ in (0,1)]
    fn=library().dcf_generate_batch
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5+[ctypes.c_int];fn.restype=ctypes.c_int
    assert fn(bits,dim,n,alpha.ctypes.data,beta.ctypes.data,roots.ctypes.data,keys[0].ctypes.data,keys[1].ctypes.data,group_bits)==0
    return keys


def evaluate_batch(key,x,party,bits,dim,group_bits=64):
    x=np.ascontiguousarray(x,dtype=np.uint64);key=np.ascontiguousarray(key,dtype=np.uint8);n=len(x)
    assert key.shape==(n,key_size(bits,dim,group_bits)) and np.all(x<(1<<bits))
    result=np.empty((n,dim),dtype=np.uint64)
    fn=library().dcf_eval_batch
    fn.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*3+[ctypes.c_int];fn.restype=ctypes.c_int
    assert fn(bits,dim,n,party,key.ctypes.data,x.ctypes.data,result.ctypes.data,group_bits)==0
    return result
