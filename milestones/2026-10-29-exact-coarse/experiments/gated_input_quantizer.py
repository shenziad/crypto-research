"""Standard offset-FSS rounding and clipping, a common gate input prefix.

Input: ring32 shares of signed fixed8 integers in units of public sa/sb.
Output: shares of clip(floor((signed_input+128)/256), -M, M), ring32.
Ties round toward +infinity. This differs from torch ties-to-even quantization.
Uses existing DCF/offset piecewise-affine techniques, not a new primitive.
"""
import numpy as np
from fused_gated_lut import random64, split64
from dcf_aes_native import generate_batch, evaluate_batch

MASK=np.uint64((1<<32)-1)


def plain_quantize(x, maximum):
    signed=((np.asarray(x,dtype=np.uint64)&MASK)^np.uint64(1<<31)).astype(np.int64)-(1<<31)
    return np.clip((signed+128)//256,-np.asarray(maximum),np.asarray(maximum)).astype(np.int64)


def prepare_round(n):
    r=random64((n,))&MASK;rs=split64(r,32)
    # floor((sint32(x)+128)/256): low correction may be -1, 0, or +1.
    const=-(r>>np.uint64(8)) - np.where(r<np.uint64(1<<31),np.uint64(1<<24),np.uint64(0))
    const+=((r&np.uint64(255))<128).astype(np.uint64)
    cs=split64(const,32)
    low=generate_batch((r+np.uint64(128))&np.uint64(255),np.full((n,1),(1<<32)-1,dtype=np.uint64),8,32)
    high=generate_batch((r+np.uint64(1<<31))&MASK,np.full((n,1),1<<24,dtype=np.uint64),32,32)
    keys=[dict(q_mask=rs[p],q_const=cs[p],q_low=low[p],q_high=high[p]) for p in (0,1)]
    return keys


def prepare(n, maximum, output_bits=32):
    assert output_bits in (32,64)
    maximum=np.broadcast_to(np.asarray(maximum,dtype=np.uint64),(n,)).copy()
    assert np.all((maximum>0)&(maximum<(1<<30)))
    keys=prepare_round(n)
    # Periodic signed clamp has exactly three coefficient jumps. Its linear
    # branches across signed zero coincide in ring32, so no fourth DCF is needed.
    r=random64((n,))&MASK;rs=split64(r,32)
    z=(MASK-r)&MASK;signed=(z^np.uint64(1<<31)).astype(np.int64)-(1<<31)
    linear=(signed>=-maximum.astype(np.int64))&(signed<maximum.astype(np.int64))
    coefficient=np.stack([linear.astype(np.uint64),np.where(linear,-r,
        np.where(signed>=0,maximum,-maximum))],axis=1)&MASK
    # Values below a masked breakpoint receive minus the forward jump.
    thresholds=((r+maximum)&MASK,(r+np.uint64(1<<31))&MASK,(r-maximum)&MASK)
    payloads=(np.stack([np.ones(n,dtype=np.uint64),-maximum-r],axis=1)&MASK,
              np.stack([np.zeros(n,dtype=np.uint64),2*maximum],axis=1)&MASK,
              np.stack([np.full(n,(1<<32)-1,dtype=np.uint64),r-maximum],axis=1)&MASK)
    if output_bits==64:
        # Lift the finite clamp directly; merely widening the modular32
        # coefficients would give wrong signs after a masked wrap.
        def wide_coeff(h):
            z=(h-r)&MASK;signed=(z^np.uint64(1<<31)).astype(np.int64)-(1<<31)
            linear=(signed>=-maximum.astype(np.int64))&(signed<maximum.astype(np.int64))
            c=np.where(linear,signed-h.astype(np.int64),np.clip(signed,-maximum.astype(np.int64),maximum.astype(np.int64)))
            return np.stack([linear.astype(np.uint64),c.view(np.uint64)],axis=1)
        coefficient=wide_coeff(np.full(n,MASK,dtype=np.uint64))
        payloads=tuple(wide_coeff((t-np.uint64(1))&MASK)-wide_coeff(t) for t in thresholds)
    coeffshares=split64(coefficient,output_bits)
    dcfs=[generate_batch(t,beta,32,output_bits) for t,beta in zip(thresholds,payloads)]
    for p in (0,1):
        keys[p].update(c_mask=rs[p],c_coeff=coeffshares[p])
        for j in range(3):keys[p]['c_dcf'+str(j)]=dcfs[j][p]
    return keys


def online(party, channel, shares, key):
    masked=(np.asarray(shares,dtype=np.uint64)+key['q_mask'])&MASK
    peer=channel.swap(masked.astype('<u4').tobytes())
    opened=(masked+np.frombuffer(peer,dtype='<u4'))&MASK
    low=evaluate_batch(key['q_low'],opened&np.uint64(255),party,8,1,32)[:,0]
    high=evaluate_batch(key['q_high'],opened,party,32,1,32)[:,0]
    q=key['q_const'].astype(np.uint64)+low+high
    if party==0:q+=opened>>np.uint64(8)
    masked=(q+key['c_mask'])&MASK
    peer=channel.swap(masked.astype('<u4').tobytes())
    opened=(masked+np.frombuffer(peer,dtype='<u4'))&MASK
    if 'c_coeff_wide' in key:
        n=len(key['c_coeff']);parts=[]
        for sl,suffix,bits in ((slice(None,n),'',32),(slice(n,None),'_wide',64)):
            coeff=key['c_coeff'+suffix].astype(np.uint64).copy();h=opened[sl]
            for j in range(3):coeff+=evaluate_batch(key['c_dcf'+str(j)+suffix],h,party,32,2,bits)
            parts.append((coeff[:,0]*h+coeff[:,1])&np.uint64((1<<bits)-1))
        return np.concatenate(parts)
    output_bits=key['c_coeff'].dtype.itemsize*8
    coeff=key['c_coeff'].astype(np.uint64).copy()
    for j in range(3):coeff+=evaluate_batch(key['c_dcf'+str(j)],opened,party,32,2,output_bits)
    return (coeff[:,0]*opened+coeff[:,1])&np.uint64((1<<output_bits)-1)


def prepare_pair(n):
    """Batch gate32/up64 quantizers into the same two opening exchanges."""
    a=prepare(n,127,32);b=prepare(n,32767,64);keys=[]
    for p in (0,1):
        key={field:np.concatenate([a[p][field],b[p][field]],axis=0)
             for field in ('q_mask','q_const','q_low','q_high','c_mask')}
        for field in ('c_coeff','c_dcf0','c_dcf1','c_dcf2'):
            key[field]=a[p][field];key[field+'_wide']=b[p][field]
        keys.append(key)
    return keys
