"""One fixed integer realization supporting bounded private output sampling.

No precision search. A/B/input use fraction16; 16 positive tanh segments use
fraction20. Every dot is exact int64 followed by signed floor. Domain is public
||h/s||2<=20, |A|,|B|<=1; the output is always within [-1,1].
"""
import numpy as np

FRACTION=16
SCALE=1<<FRACTION
OUTPUT_FRACTION=20
OUTPUT_SCALE=1<<OUTPUT_FRACTION
INPUT_L2_CAP=20
TANH_NODES=np.array([0,256816,484565,666002,798589,889490,949117,
    987104,1010856,1025535,1034540,1040040,1043391,1045428,1046665,
    1047417,1047873],dtype=np.int64)


def encode_parameters(a,b):
    assert a.shape[1]==b.shape[0]==8 and b.shape[1]==128
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.max(np.abs(a))<=1 and np.max(np.abs(b))<=1
    return np.rint(a.astype(np.float64)*SCALE).astype(np.int64),np.rint(b.astype(np.float64)*SCALE).astype(np.int64)


def tanh_integer(z):
    magnitude=np.abs(z)
    segment=np.minimum(magnitude>>14,15)
    remainder=magnitude & ((1<<14)-1)
    interpolated=TANH_NODES[segment]+((TANH_NODES[segment+1]-TANH_NODES[segment])*remainder)//(1<<14)
    result=np.where(magnitude>=4*SCALE,OUTPUT_SCALE,interpolated)
    result=np.where(z<0,-result,result)
    assert np.max(np.abs(result))<=OUTPUT_SCALE
    return result


def evaluate(x,ai,bi,chunk=32):
    assert np.isfinite(x).all() and np.max(np.linalg.norm(x,axis=-1))<=INPUT_L2_CAP
    xi=np.rint(x*SCALE).astype(np.int64)
    t_raw=xi@bi.T
    ti=t_raw//SCALE
    result=np.empty((len(x),len(ai)),dtype=np.int64)
    stats=dict(max_abs_input_integer=int(np.abs(xi).max()),
        max_abs_t_accumulator=int(np.abs(t_raw).max()),
        max_abs_t_integer=int(np.abs(ti).max()),max_abs_z_accumulator=0,max_abs_z_integer=0)
    for start in range(0,len(x),chunk):
        raw=ti[start:start+chunk]@ai.T
        z=raw//SCALE
        stats['max_abs_z_accumulator']=max(stats['max_abs_z_accumulator'],int(np.abs(raw).max()))
        stats['max_abs_z_integer']=max(stats['max_abs_z_integer'],int(np.abs(z).max()))
        result[start:start+chunk]=tanh_integer(z)
    return result,stats
