"""Public sparse cyclic convolution of a standard DCF; group64 payloads."""
import ctypes
from pathlib import Path
import numpy as np
from dcf_aes_native import key_size

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_sparse_native.so'
_lib=None


def derivative(table,bits=32,shift=8,maximum=127):
    table=np.asarray(table,dtype=np.int64);assert table.shape==(256,)
    ring=1<<bits;assert maximum*(1<<shift)<ring//2
    codes=np.arange(-maximum+1,maximum+1,dtype=np.int64)
    thresholds=(codes*(1<<shift)-(1<<(shift-1)))%ring
    deltas=table[codes&255]-table[(codes-1)&255]
    thresholds=np.append(thresholds,ring//2)
    deltas=np.append(deltas,table[(-maximum)&255]-table[maximum])
    nz=deltas!=0;thresholds=thresholds[nz].astype(np.uint64);deltas=deltas[nz]
    order=np.argsort(thresholds);thresholds=thresholds[order];deltas=deltas[order]
    assert int(deltas.sum())==0
    return thresholds,deltas,int(table[0])


def weighted(key,opened,thresholds,weights,party,bits=32,dim=2,cached=True):
    global _lib
    key=np.ascontiguousarray(key,dtype=np.uint8);opened=np.ascontiguousarray(opened,dtype=np.uint64)
    thresholds=np.ascontiguousarray(thresholds,dtype=np.uint64);weights=np.ascontiguousarray(weights,dtype=np.uint64)
    n=len(opened);assert key.shape==(n,key_size(bits,dim,64)) and len(thresholds)==len(weights)>0
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));_lib.dcf_sparse_weighted.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*5
        _lib.dcf_sparse_weighted.restype=ctypes.c_int
    out=np.empty((n,dim),dtype=np.uint64)
    assert _lib.dcf_sparse_weighted(bits,dim,n,party,len(thresholds),int(cached),key.ctypes.data,opened.ctypes.data,
        thresholds.ctypes.data,weights.ctypes.data,out.ctypes.data)==0
    return out


def public_constant(opened,thresholds,deltas,f0):
    # Public comparisons, no secret-dependent shape. Split large n if needed.
    return (np.int64(f0)-((opened[:,None]<thresholds[None,:])*deltas[None,:]).sum(axis=1)).view(np.uint64)
