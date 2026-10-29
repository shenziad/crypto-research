"""C0 integer SiLU and certified coarse-classifier/fine-body interface."""
import math
import numpy as np
from public_silu_spline import KNOTS,silu


def build(sa,shift,fraction_bits=22,input_bits=33,input_fraction_bits=8):
    assert 1<=shift<=8 and input_bits>=17 and sa>0;bucket=1<<shift;ring=1<<input_bits
    assert input_fraction_bits in (4,8);scale=1<<input_fraction_bits
    lower,upper=-127*scale,127*scale
    knots=sorted(set([lower,upper]+[int(round(x*scale/(sa*bucket)))*bucket for x in KNOTS if lower<x*scale/sa<upper]))
    knots=[v for v in knots if lower<=v<=upper];assert all(v%bucket==0 for v in knots)
    values=[int(round(silu(x*sa/scale)*(1<<fraction_bits))) for x in knots]
    starts=[-(ring//2),*knots];ends=[*knots,ring//2];slopes=[0];intercepts=[values[0]];current=values[0]
    for a,b,fb in zip(knots[:-1],knots[1:],values[1:]):
        m=int(round((fb-current)/(b-a)));slopes.append(m);intercepts.append(current-m*a);current+=m*(b-a)
    slopes.append(0);intercepts.append(current)
    unsigned=[];extrema=[];boundary_errors=[]
    for i,(a,b,m,c) in enumerate(zip(starts,ends,slopes,intercepts)):
        extrema.extend([m*a+c,m*(b-1)+c])
        if a<0:unsigned.append((a+ring,min(b,0)+ring,m,c-m*ring))
        if b>0:unsigned.append((max(a,0),b,m,c))
        if i:
            assert slopes[i-1]*a+intercepts[i-1]==m*a+c
            dm=m-slopes[i-1];error=abs(dm)*(bucket-1)
            boundary_errors.append(dict(knot=a,delta_slope=dm,maximum_integer_error=error,maximum_expected_bias=abs(dm)*bucket/4))
            # Next-piece extrapolation over previous bucket is the only
            # possible location error for aligned nodes and this dither.
            extrema.extend([m*(a-bucket+1)+c,m*(a-1)+c])
    unsigned.sort();assert all(a%bucket==0 and b%bucket==0 for a,b,_,_ in unsigned)
    assert all(unsigned[i][1]==unsigned[i+1][0] for i in range(len(unsigned)-1))
    return dict(sa=float(sa),fraction_bits=fraction_bits,input_bits=input_bits,input_fraction_bits=input_fraction_bits,classification_shift=shift,
        classifier_bits=input_bits-shift,promised_input_bits=input_bits-1,knots=knots,
        signed_starts=starts,signed_ends=ends,slopes=slopes,intercepts=intercepts,unsigned_pieces=unsigned,
        continuous_integer_pieces=True,absolute_bound=max(abs(v) for v in extrema),
        boundary_error_certificate=boundary_errors,maximum_classifier_integer_error=max(v['maximum_integer_error'] for v in boundary_errors),
        definition='AlignedC0 integer-affine function. Public dither selects coarse interval; polynomial still uses original fine input. Signed input has one guard bit in classifier view.')


def evaluate(spec,x,rho=None):
    x=np.asarray(x,dtype=np.int64);grid=1<<spec['classification_shift']
    if rho is None:query=x
    else:
        rho=np.asarray(rho,dtype=np.int64);assert np.all(rho>=0) and np.all(rho<grid)
        promise=spec['promised_input_bits'];assert np.all(x>=-(1<<(promise-1))) and np.all(x<(1<<(promise-1)))
        query=(x//grid+(rho<(x%grid)).astype(np.int64))*grid
    index=np.searchsorted(np.asarray(spec['signed_starts'][1:],dtype=np.int64),query,side='right')
    return np.asarray(spec['slopes'],dtype=np.int64)[index]*x+np.asarray(spec['intercepts'],dtype=np.int64)[index]
