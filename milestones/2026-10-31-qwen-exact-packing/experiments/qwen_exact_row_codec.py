"""Fixed bijective XOR-linear packing of8 signed18 coefficient bit patterns.

No precision change. 8*18=144 bits, exactly18 bytes; each input bit retains
one unique output position, so packing and unpacking commute with XOR shares.
"""
import numpy as np


def pack18(coefficients):
    a=np.asarray(coefficients,dtype=np.int64)
    assert a.ndim==2 and a.shape[1]==8
    words=(a&((1<<18)-1)).astype(np.uint32)
    result=np.zeros((len(a),18),dtype=np.uint8)
    for j in range(8):
        byte,shift=divmod(18*j,8)
        shifted=words[:,j]<<shift
        for k in range(3):
            result[:,byte+k]|=((shifted>>(8*k))&255).astype(np.uint8)
    return result


def unpack18(packed):
    a=np.asarray(packed,dtype=np.uint8)
    assert a.ndim==2 and a.shape[1]==18
    result=np.empty((len(a),8),dtype=np.uint32)
    for j in range(8):
        byte,shift=divmod(18*j,8)
        word=(a[:,byte].astype(np.uint32)|(a[:,byte+1].astype(np.uint32)<<8)|
              (a[:,byte+2].astype(np.uint32)<<16))
        result[:,j]=(word>>shift)&((1<<18)-1)
    return result


def signed18(words):
    a=np.asarray(words,dtype=np.int64)
    assert np.all((a>=0)&(a<(1<<18)))
    return np.where(a&(1<<17),a-(1<<18),a)
