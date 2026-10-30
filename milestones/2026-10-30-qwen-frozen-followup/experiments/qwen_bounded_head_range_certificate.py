"""Conditional exact ranges for frozen Qwen head; same F16/F20 definition.

Public model-class norm caps are additional declared parameters. Numerical
certificate only: no private input validation, protocol timing or security proof.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from bounded_adapter_fixed16 import SCALE,OUTPUT_SCALE

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rep(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))
def ceil(x):return -(-x.numerator//x.denominator)


def main():
    destination=ROOT/'results/qwen_bounded_head_range_certificate.json'
    assert not destination.exists()
    pilot=json.loads((ROOT/'results/bounded_adapter_qwen_transfer.json').read_bytes())
    path=ROOT/'results/bounded_adapter_qwen_transfer_weights.npz'
    assert sha(path)==pilot['weights_sha256']
    with np.load(path) as z:
        a=z['bounded_A'];b=z['bounded_B'];scale=F.from_float(float(z['hidden_scale']))
    assert a.shape==(151936,8) and b.shape==(8,896)
    ai=np.rint(a.astype(np.float64)*SCALE).astype(np.int64)
    bi=np.rint(b.astype(np.float64)*SCALE).astype(np.int64)
    # Float32 products are exact in float64; positive sums have standard
    # gamma_n bounds regardless of reduction order. Compare exact rational
    # upper bounds rather than treating rounded norms as exact certificates.
    u=F(1,1<<53)
    gamma=lambda n:n*u/(1-n*u)
    la=F(3,2);rb=F(9,4);D=F(30);H=F(52);S=SCALE;r=8;d=896
    actual_a=F.from_float(float(np.abs(a.astype(np.float64)).sum(1).max()))/(1-gamma(7))
    actual_b2=F.from_float(float(np.square(b.astype(np.float64)).sum(1).max()))/(1-gamma(895))
    assert actual_a<=la and actual_b2<=rb*rb
    assert int(np.abs(ai).sum(1).max())<=la*S
    assert int(np.square(bi).sum(1).max())<=rb*rb*S*S
    assert abs(a.astype(np.float64)-ai/S).max()<=1/(2*S)
    assert abs(b.astype(np.float64)-bi/S).max()<=1/(2*S)
    assert D*D>d and max(float(abs(a).max()),float(abs(b).max()))<.5
    # Ideal real RMSNorm: ||h/scale||<=sqrt(d)*max|gamma|/scale.
    model=ROOT/'models/Qwen2.5-0.5B/model.safetensors'
    with model.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(length))
        entry=meta['model.norm.weight'];f.seek(8+length+entry['data_offsets'][0])
        raw=f.read(entry['data_offsets'][1]-entry['data_offsets'][0])
    assert entry['dtype']=='BF16' and entry['shape']==[896]
    norm=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4')
    normbound=F.from_float(float(abs(norm).max()))*D/scale
    assert normbound<H
    et=D*H/(2*S)+rb*D/(2*S)+F(1,S)
    tref=rb*H
    ez=r*tref/(2*S)+la*et+F(1,S)
    pwl=F(77,12800)+F(3,2*OUTPUT_SCALE)
    # Reuse the unchanged node/curvature/saturation proof, not a precision search.
    old=ROOT/'results/bounded_adapter_fixed16_certificate.json'
    assert json.loads(old.read_bytes())['all_passed']
    rawt=ceil(S*S*rb*(H+D/(2*S)));tmax=ceil(F(rawt,S))+1
    rawz=ceil(S*la*tmax);zmax=ceil(F(rawz,S))+1
    limits=dict(max_abs_input_integer=ceil(S*H+F(1,2)),
        max_abs_t_accumulator=rawt,max_abs_t_integer=tmax,
        max_abs_z_accumulator=rawz,max_abs_z_integer=zmax)
    widths=dict(max_abs_input_integer=23,max_abs_t_accumulator=41,
        max_abs_t_integer=25,max_abs_z_accumulator=44,max_abs_z_integer=28)
    for name,bits in widths.items():assert limits[name]<(1<<(bits-1))
    with np.load(ROOT/'results/bounded_adapter_qwen_frozen_captures.npz') as cap:
        for corpus in ('ptb_test','wiki_test'):
            x=cap[corpus+'_x'].astype(np.float64)
            assert float(np.linalg.norm(x))<float(H)
    out=dict(all_passed=True,domain=dict(hidden=896,vocab=151936,rank=8,
        input_l2_cap=52,coefficient_abs_cap=.5,B_row_l2_cap=rep(rb),A_row_l1_cap=rep(la)),
        public_norm_caps_are_additional_interface_parameters=True,
        norm_caps_disclose_coarse_bounds=True,real_ideal_RMSNorm_l2_upper=rep(normbound),
        observed_float_A_row_l1_certified_upper=rep(actual_a),
        observed_float_B_row_squared_l2_certified_upper=rep(actual_b2),
        integer_range_bounds=limits,signed_bits_sufficient=widths,
        error_bounds=dict(t_infinity=rep(et),z_infinity=rep(ez),
            pwl=rep(pwl),total_delta_uniform=rep(ez+pwl),NLL_change_uniform=rep(2*(ez+pwl))),
        fractions=dict(input_weights=16,score=20),
        full_sampler_requirements=dict(alias_padded_rows=262144,secret_index_bits=18,
            alias_precision=44,q_integer_total_bits=62,E31_output_bits=33,
            max_normalizer_power=93,signed_CDF_bits=95,joint_CDF_uniform_bits=64,
            dense_CDF_TV_bound=rep(F(151936,1<<64)),fixed_rejection_trials=201),
        weights_sha256=sha(path),model_sha256=pilot['model_sha256'],
        unchanged_PWL_certificate_sha256=sha(old),source_sha256=sha(Path(__file__)),
        limits='Conditional analytical correctness/range only. Public inputL2/BrowL2/ArowL1 caps must be explicit in ideal functionality; they are not secret statistics newly released by a protocol. Exact stored-float norms checked with conservative gamma_n arithmetic. Ideal real RMSNorm bound does not prove all floating CUDA executions; sampled inputs checked separately. SameF16/F20, no training/precision search. Currentnative128/50304 implementation has NOT been ported: xdecode needs23bits, indices/output need18bits, p44 qmass total62bits/CDF95. No full Qwen private performance, joint security, malicious input verification orCCFclaim.')
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('all_passed','integer_range_bounds','error_bounds','full_sampler_requirements')},indent=2))


if __name__=='__main__':main()
