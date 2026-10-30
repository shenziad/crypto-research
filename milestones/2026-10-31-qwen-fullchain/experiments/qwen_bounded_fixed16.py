"""Qwen896/rank8 fixed F16/F20 reference, conditional on the new norm caps."""
import numpy as np
from bounded_adapter_fixed16 import SCALE,OUTPUT_SCALE,tanh_integer


def encode_parameters(a,b):
    assert a.shape==(151936,8) and b.shape==(8,896)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert max(float(abs(a).max()),float(abs(b).max()))<=.5
    return np.rint(a.astype(np.float64)*SCALE).astype(np.int64),np.rint(b.astype(np.float64)*SCALE).astype(np.int64)


def evaluate(x,ai,bi,chunk=16):
    assert x.shape[1]==896 and np.isfinite(x).all()
    assert np.max(np.linalg.norm(x,axis=-1))<=52
    xi=np.rint(x*SCALE).astype(np.int64)
    rt=xi@bi.T;t=rt//SCALE
    output=np.empty((len(x),len(ai)),dtype=np.int64)
    stats=dict(max_abs_input_integer=int(abs(xi).max()),
        max_abs_t_accumulator=int(abs(rt).max()),max_abs_t_integer=int(abs(t).max()),
        max_abs_z_accumulator=0,max_abs_z_integer=0)
    for start in range(0,len(x),chunk):
        raw=t[start:start+chunk]@ai.T;z=raw//SCALE
        stats['max_abs_z_accumulator']=max(stats['max_abs_z_accumulator'],int(abs(raw).max()))
        stats['max_abs_z_integer']=max(stats['max_abs_z_integer'],int(abs(z).max()))
        output[start:start+chunk]=tanh_integer(z)
    return output,stats
