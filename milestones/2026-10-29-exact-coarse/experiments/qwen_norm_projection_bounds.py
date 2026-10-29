"""Exact dyadic upper bounds from the pre-MLP RMSNorm mathematical contract.

No claim that floating/private RMSNorm implementations satisfy the norm contract
without an error proof. Bounds become conditional public fixed-point input types.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]


def ceil_sqrt(q):
    t=math.isqrt(q.numerator//q.denominator)
    return t if t*t*q.denominator==q.numerator else t+1


def main():
    torch.set_num_threads(4)
    ranges=json.loads((ROOT/'results/qwen_gate_ranges.json').read_text())
    # Use exactly the same CUDA float32 scale construction as model experiments.
    am=torch.tensor(ranges['gate_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values/127
    bm=torch.tensor(ranges['up_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values/32767
    sa=am.cpu().numpy();sb=bm.cpu().numpy();rows=[]
    path=ROOT/'models/Qwen2.5-0.5B/model.safetensors'
    with safe_open(path,framework='pt',device='cpu') as model:
        names=set(model.keys())
        for layer in range(24):
            gamma=model.get_tensor(f'model.layers.{layer}.post_attention_layernorm.weight').float().numpy().astype(np.float64)
            d=len(gamma)
            for projection,scale in (('gate_proj',float(sa[layer])),('up_proj',float(sb[layer]))):
                name=f'model.layers.{layer}.mlp.{projection}'
                assert name+'.bias' not in names
                w=model.get_tensor(name+'.weight').float().numpy().astype(np.float64)
                coefficient=w*gamma[None,:] # product of two float32 dyadics is exact in float64
                peak=float(np.abs(coefficient).max());frac=min(24,math.floor(math.log2(math.isqrt((1<<62)//d)/max(peak,1e-300))))
                assert frac>=0
                upper=np.ceil(np.abs(coefficient)*(1<<frac)).astype(np.int64)
                assert int(upper.max())**2*d<(1<<62)
                squared=(upper*upper).sum(axis=1,dtype=np.int64);S=int(squared.max())
                scale_q=Fraction.from_float(scale)
                bounds={}
                for budget in (1,2):
                    squared_encoded=Fraction((256*budget)**2*d*S,1<<(2*frac))/(scale_q*scale_q)
                    bound=ceil_sqrt(squared_encoded)
                    bits=bound.bit_length()+1
                    bits_with_halfup_gap=(bound+128).bit_length()+1
                    assert bound < 1<<(bits-1)
                    assert bound+128 < 1<<(bits_with_halfup_gap-1)
                    bounds[str(budget)]=dict(encoded_abs_bound=bound,signed_input_bits=bits,
                        input_bits_with_halfup_gap=bits_with_halfup_gap,
                        exact_round_reduce_output_bits=bits_with_halfup_gap-8)
                rows.append(dict(layer=layer,projection=projection,hidden_dimension=d,public_scale=scale,
                    coefficient_upper_fraction_bits=frac,maximum_row_sum_upper_integer_squares=S,bounds=bounds))
            print('layer',layer,'bounds ready',flush=True)
    out=dict(layers=rows,all_passed=True,range_sha256=hashlib.sha256((ROOT/'results/qwen_gate_ranges.json').read_bytes()).hexdigest(),
        mathematical_contract='v=x/sqrt(mean(x^2)+epsilon), epsilon>=0 => ||v||2<=sqrt(d); weighted projection uses exact coefficients W*gamma. Budget2 is an assumed enlarged norm budget, not a proved floating-point error bound.',
        exactness='Dyadic coefficient products exact in float64. Ceil-to-integer bounds, square sums and comparison to norm/type thresholds exact integers/Fractions. Public norm bounds are fixed per model/layer, not selected on private test activations.',
        limits='Conditional mathematical-model input range certificate. Finite private/FP32 RMSNorm and upstream projection/normalization conversion still need their own error/overflow proof. Does not authorize silently restricting previous full-signed32 test functionality.')
    (ROOT/'results/qwen_norm_projection_bounds.json').write_text(json.dumps(out,indent=2)+'\n')
    for p in ('gate_proj','up_proj'):
        print(p,'budget2 bits',[v['bounds']['2']['input_bits_with_halfup_gap'] for v in rows if v['projection']==p],flush=True)


if __name__=='__main__':main()
