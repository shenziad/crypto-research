"""Torch-tensor/ctypes adapter for CUDA DCF research backend."""
import ctypes
from pathlib import Path
import numpy as np
import torch

LIB=Path(__file__).resolve().parents[1]/'toolchain/dcf_cuda.so'
PROTOCOL_LIB=LIB.with_name('dcf_cuda_protocol.so');_libs={}


def tables():
    def mul(a,b):
        v=0
        for _ in range(8):
            if b&1:v^=a
            a=((a<<1)^(0x11b if a&128 else 0))&255;b>>=1
        return v
    def power(x,n):
        y=1
        while n:
            if n&1:y=mul(y,x)
            x=mul(x,x);n>>=1
        return y
    def rot(x,n):return ((x<<n)|(x>>(8-n)))&255
    sb=[]
    for x in range(256):
        a=power(x,254) if x else 0
        sb.append(a^rot(a,1)^rot(a,2)^rot(a,3)^rot(a,4)^0x63)
    assert sb[0]==0x63 and sb[0x53]==0xed
    te=[(mul(x,2)<<24)|(x<<16)|(x<<8)|mul(x,3) for x in sb]
    return np.asarray(sb,dtype=np.uint8),np.asarray(te,dtype=np.uint32)


def library(protocol=False):
    if protocol not in _libs:
        torch.cuda.init();_lib=ctypes.CDLL(str(PROTOCOL_LIB if protocol else LIB))
        _lib.dcf_cuda_init.argtypes=[ctypes.c_void_p]*2;_lib.dcf_cuda_init.restype=ctypes.c_int
        sb,te=tables();assert _lib.dcf_cuda_init(sb.ctypes.data,te.ctypes.data)==0
        _lib.dcf_cuda_select.argtypes=[ctypes.c_int]*7+[ctypes.c_void_p]*5;_lib.dcf_cuda_select.restype=ctypes.c_int
        _lib.aes_cuda_eval.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4;_lib.aes_cuda_eval.restype=ctypes.c_int
        if protocol:
            _lib.dcf_cuda_select_typed.argtypes=[ctypes.c_int]*8+[ctypes.c_void_p]*5;_lib.dcf_cuda_select_typed.restype=ctypes.c_int
            _lib.cuda_private_contraction.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*5+[ctypes.c_int]+[ctypes.c_void_p]*2;_lib.cuda_private_contraction.restype=ctypes.c_int
        _libs[protocol]=_lib
    return _libs[protocol]


def tensor(a):
    a=np.ascontiguousarray(a)
    if a.dtype==np.uint64:a=a.view(np.int64)
    return torch.from_numpy(a).to('cuda')


def aes(keys,plain):
    assert keys.shape==plain.shape and keys.dtype==plain.dtype==torch.uint8 and keys.shape[1]==16
    output=torch.empty_like(plain);lib=library()
    assert lib.aes_cuda_eval(len(keys),keys.data_ptr(),plain.data_ptr(),output.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
    return output


def select(keys,x,coords,party,bits,dim,group_bits=64,protocol=False):
    protocol=protocol or group_bits==32;lib=library(protocol);n=len(keys)
    assert group_bits in (32,64) and keys.dtype==torch.uint8 and keys.shape[1]==16+bits*(17+(group_bits//8)*dim)+(group_bits//8)*dim
    assert x.dtype==torch.int64 and coords.dtype==torch.int32
    x=x.reshape(n,-1);queries=x.shape[1];per_query=int(coords.ndim>1);count=coords.shape[-1]
    assert keys.is_contiguous() and x.is_contiguous() and coords.is_contiguous() and 1<=count<=8
    assert keys.is_cuda and x.is_cuda and coords.is_cuda
    if per_query:assert coords.numel()==n*queries*count
    output=torch.empty((n,queries,count),device='cuda',dtype=torch.int64)
    args=(bits,dim,n,queries,party,count,per_query)
    if protocol:args=args+(group_bits,)
    fn=lib.dcf_cuda_select_typed if protocol else lib.dcf_cuda_select
    assert fn(*args,keys.data_ptr(),x.data_ptr(),coords.data_ptr(),output.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
    return output


def contract(public_w,mask_w,mask_v,zhat,product,party):
    m,n=public_w.shape;out=torch.empty(m,dtype=torch.int64,device='cuda')
    assert mask_w.shape==(m,n) and mask_v.numel()==zhat.numel()==n and product.numel()==m
    assert all(v.is_cuda and v.is_contiguous() and v.dtype==torch.int64 for v in (public_w,mask_w,mask_v,zhat,product))
    assert library(True).cuda_private_contraction(m,n,public_w.data_ptr(),mask_w.data_ptr(),mask_v.data_ptr(),zhat.data_ptr(),product.data_ptr(),party,out.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
    return out
