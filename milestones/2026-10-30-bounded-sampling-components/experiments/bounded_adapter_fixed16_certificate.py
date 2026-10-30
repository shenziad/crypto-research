"""Exact rational coefficient, range and uniform-error certificate.

Conditional on the declared real normalized-input domain; not a verification
of all floating-point executions of the public backbone or a 2PC proof.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from bounded_adapter_fixed16 import TANH_NODES,SCALE,OUTPUT_SCALE,encode_parameters

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rep(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))
def ceil(x):return -(-x.numerator//x.denominator)


def load_norm():
    path=ROOT/'models/Pythia-14M/model.safetensors'
    out={}
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(length));base=8+length
        for name in ('weight','bias'):
            v=meta['gpt_neox.final_layer_norm.'+name]
            f.seek(base+v['data_offsets'][0]);raw=f.read(v['data_offsets'][1]-v['data_offsets'][0])
            if v['dtype']=='F16':a=np.frombuffer(raw,dtype='<f2').astype(np.float32)
            elif v['dtype']=='BF16':a=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4')
            elif v['dtype']=='F32':a=np.frombuffer(raw,dtype='<f4')
            else:raise ValueError(v['dtype'])
            assert list(a.shape)==v['shape']==[128]
            out[name]=[F.from_float(float(t)) for t in a]
    return out


def main():
    destination=ROOT/'results/bounded_adapter_fixed16_certificate.json';assert not destination.exists()
    nodes=[]
    for k,value in enumerate(TANH_NODES.tolist()):
        x=F(k,2);term=F(1);lo=term
        for j in range(1,129):term*=x/j;lo+=term
        hi=lo+(term*x/129)/(1-x/130)
        a=(lo-1)/(lo+1);b=(hi-1)/(hi+1)
        assert F(value)-F(1,2)<a*OUTPUT_SCALE and b*OUTPUT_SCALE<F(value)+F(1,2)
        nodes.append(dict(k=k,integer=value,rounding_enclosure_verified=True))
    # sqrt128 < 11.314, max|tanh''| < .77, by squaring positive rationals.
    D=F(5657,500);assert D*D>128 and 77**2*27>160000
    H=20;r=8;d=128;S=SCALE
    t_reference=D*H
    et=D*H/(2*S)+F(d,2*S)+F(1,S)
    ez=r*t_reference/(2*S)+r*et+F(1,S)
    interpolation=F(77,12800)+F(3,2*OUTPUT_SCALE)
    # e8>1999, hence the |z|>=4 saturation discrepancy is below.001.
    lower=sum((F(8)**k/F(__import__('math').factorial(k)) for k in range(11)),F(0))
    assert lower>1999 and F(1,1000)<interpolation
    eta=ez+interpolation
    raw_t=ceil(S*S*D*(H+D/(2*S)))
    ti=ceil(F(raw_t,S))+1
    raw_z=r*S*ti
    zi=ceil(F(raw_z,S))+1
    limits=dict(max_abs_input_integer=ceil(S*H+F(1,2)),max_abs_t_accumulator=raw_t,
                max_abs_t_integer=ti,max_abs_z_accumulator=raw_z,max_abs_z_integer=zi,
                max_abs_interpolation_product=OUTPUT_SCALE*((1<<14)-1))
    assert all(v<(1<<63) for v in limits.values())
    archive=ROOT/'results/bounded_output_adapter_weights.npz'
    with np.load(archive) as z:
        a=z['bounded_A'];b=z['bounded_B'];scale=F.from_float(float(z['hidden_scale']))
        ai,bi=encode_parameters(a,b)
    assert np.abs(a.astype(np.float64)-ai/S).max()<=1/(2*S)
    assert np.abs(b.astype(np.float64)-bi/S).max()<=1/(2*S)
    norm=load_norm();gamma=max(map(abs,norm['weight']));beta_squared=sum((x*x for x in norm['bias']),F(0))
    assert beta_squared<225
    ideal_normalized_l2_bound=(gamma*D+15)/scale
    assert ideal_normalized_l2_bound<H
    out=dict(all_passed=True,domain=dict(hidden_scaled_l2_cap=H,coefficient_abs_cap=1,d=d,rank=r),
        fractions=dict(input_weights=16,output=20),tanh_nodes=nodes,
        error_bounds=dict(t_infinity=rep(et),z_infinity=rep(ez),pwl_uniform=rep(interpolation),
                          total_delta_uniform=rep(eta),nll_change_bound=rep(2*eta)),
        integer_range_bounds=limits,ideal_public_layernorm_l2_upper=rep(ideal_normalized_l2_bound),
        parameter_rounding_entries=a.size+b.size,weights_sha256=sha(archive),
        model_sha256=sha(ROOT/'models/Pythia-14M/model.safetensors'),
        source_sha256={name:sha(ROOT/'experiments'/name) for name in
            ['bounded_adapter_fixed16.py','bounded_adapter_fixed16_certificate.py']},
        limits='Exact rational analytical bound for real ||h/s||2<=20 and A/B coordinatecaps1, integer coefficients/operations defined here. Public LayerNorm bound is for ideal real normalization, not a full floating-point implementation proof; measured GPU inputs must be checked separately. No private arithmetic circuit, speed, independent novelty, full rejection or CCF claim. Fixed16 is supporting implementation only, not a dedicated quantization mainline.')
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,eta=float(eta),nll_bound=float(2*eta),ranges=limits,
                         ideal_norm_bound=float(ideal_normalized_l2_bound)),indent=2))


if __name__=='__main__':main()
