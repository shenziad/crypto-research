"""Public bounded integer-affine SiLU specification; no cryptographic claim."""
import math
import numpy as np

KNOTS=(-8,-6,-4,-3,-2.5,-2,-1.5,-1,-.75,-.5,-.25,0,.25,.5,.75,1,1.5,2,2.5,3,4,6,8)


def silu(x):
    if x>=0:return x/(1+math.exp(-x))
    e=math.exp(x);return x*e/(1+e)


def build(sa,fraction_bits=22,input_bits=32):
    assert sa>0 and 1<=fraction_bits<=24 and 17<=input_bits<=63
    ring=1<<input_bits
    lower,upper=-127*256,127*256
    knots=sorted(set([lower,upper]+[int(round(x*256/sa)) for x in KNOTS if lower<x*256/sa<upper]))
    values=[int(round(silu(x*sa/256)*(1<<fraction_bits))) for x in knots]
    starts=[-(ring//2),*knots];ends=[*knots,ring//2]
    slopes=[0];intercepts=[values[0]]
    for a,b,fa,fb in zip(knots[:-1],knots[1:],values[:-1],values[1:]):
        m=int(round((fb-fa)/(b-a)));slopes.append(m);intercepts.append(fa-m*a)
    slopes.append(0);intercepts.append(values[-1])
    assert len(starts)==len(slopes)==len(ends)
    unsigned=[];extrema=[]
    for a,b,m,c in zip(starts,ends,slopes,intercepts):
        assert a<b
        extrema.extend([m*a+c,m*(b-1)+c])
        if a<0:unsigned.append((a+ring,min(b,0)+ring,m,c-m*ring))
        if b>0:unsigned.append((max(a,0),b,m,c))
    unsigned.sort();assert unsigned[0][0]==0 and unsigned[-1][1]==ring
    assert all(unsigned[i][1]==unsigned[i+1][0] for i in range(len(unsigned)-1))
    # Delta F(t), t=1..2^32-1, as constant-slope runs including boundary jumps.
    runs=[]
    for j,(a,b,m,c) in enumerate(unsigned):
        if a:
            old=unsigned[j-1];jump=m*a+c-(old[2]*(a-1)+old[3])
            if jump:runs.append((a,a,jump))
        lo=max(1,a+1)
        if m and lo<=b-1:runs.append((lo,b-1,m))
    merged=[]
    for a,b,s in runs:
        if merged and merged[-1][1]+1==a and merged[-1][2]==s:merged[-1]=(merged[-1][0],b,s)
        else:merged.append((a,b,s))
    f0=next(m*0+c for a,b,m,c in unsigned if a<=0<b)
    last=unsigned[-1][2]*(ring-1)+unsigned[-1][3]
    assert sum(s*(b-a+1) for a,b,s in merged)==last-f0
    return dict(sa=float(sa),fraction_bits=fraction_bits,input_bits=input_bits,signed_starts=starts,signed_ends=ends,
        slopes=slopes,intercepts=intercepts,unsigned_pieces=unsigned,delta_runs=merged,
        function_at_zero=f0,function_at_last=last,minimum=min(extrema),maximum=max(extrema),
        absolute_bound=max(abs(v) for v in extrema),clip_fixed8_bound=upper,
        definition='Integer coefficients and half-open intervals define the exact finite function; endpoints checked with Python integers.')


def evaluate(spec,x):
    bits=spec.get('input_bits',32)
    x=np.asarray(x,dtype=np.int64);assert np.all(x>=-(1<<(bits-1))) and np.all(x<(1<<(bits-1)))
    i=np.searchsorted(np.asarray(spec['signed_starts'][1:],dtype=np.int64),x,side='right')
    return np.asarray(spec['slopes'],dtype=np.int64)[i]*x+np.asarray(spec['intercepts'],dtype=np.int64)[i]
