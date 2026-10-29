"""Necessary range screen for a future full finite FFN with common RMS input.

CPU only. Fold gamma into signed16 projection weights, certify spectral norms
through exact integer Gram/Gershgorin bounds, then bound gated down contraction.
Not a certificate for arbitrary independently supplied A/B wires or FP32 ops.
"""
import json
import math
import hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
from safetensors import safe_open
from dithered_silu import build

ROOT=Path(__file__).resolve().parents[1]


def ceil_sqrt(n):
    s=math.isqrt(int(n));return s+(s*s<n)


def root_upper(n):return Fraction(ceil_sqrt(int(n)*(1<<40)),1<<20)


def quantize16(matrix,max_fraction=16):
    for fraction in range(max_fraction,0,-1):
        q=np.rint(matrix.astype(np.float64)*(1<<fraction)).astype(np.int64)
        if int(q.min())>=-32768 and int(q.max())<=32767:return q.astype(np.int16),fraction
    raise ValueError('No signed16 poweroftwo projection representation')


def norm_certificate(q,fraction):
    n,d=q.shape;qi=q.astype(np.int64);col=(qi*qi).sum(0);maximum=int(col.max())
    assert maximum<(1<<53) and d*maximum<(1<<62)
    # Every absolute Gram dot is <= maxcolumnL2² by Cauchy. Integer products
    # and allpartial absolute sums are <2^53, so this FP64 BLAS dot is exact.
    gram_float=q.astype(np.float64).T@q.astype(np.float64)
    assert np.array_equal(gram_float,np.rint(gram_float))
    gram=gram_float.astype(np.int64);assert np.array_equal(gram.diagonal(),col)
    radius=int(np.abs(gram).sum(1).max());numerator=ceil_sqrt(radius)
    return Fraction(numerator,1<<fraction),dict(shape=[n,d],weight_fraction_bits=fraction,
        quantized_matrix_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        max_column_integer_l2_squared=maximum,integer_gram_exact_by_absolute_dot_bound=True,
        gershgorin_gram_row_absolute_sum=radius,norm_upper_numerator=numerator,norm_upper_denominator=1<<fraction)


def gate_linear_bound(spec):
    sa=Fraction.from_float(spec['sa']);fraction=spec['fraction_bits'];target=sa*(1<<fraction)/256
    gain=(target.numerator+target.denominator-1)//target.denominator;constant=0
    for a,b,m,c in zip(spec['signed_starts'],spec['signed_ends'],spec['slopes'],spec['intercepts']):
        candidates=[a,b-1]
        if a<=0<b:candidates.append(0)
        for x in candidates:constant=max(constant,abs(m*x+c)-gain*abs(x))
    assert constant>=0
    return gain,constant


def main():
    rows=[]
    for model,folder,layers,ranges_name in [('qwen05','Qwen2.5-0.5B',(0,12,23),'qwen_gate_ranges.json'),('qwen15','Qwen2.5-1.5B',(0,14,27),'qwen15_gate_ranges.json')]:
        calibration=json.loads((ROOT/'results'/ranges_name).read_text())
        if model=='qwen05':
            old=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());sa=[s['sa'] for s in old['specifications']]
            sb=[float(np.float32(max(v))/np.float32(32767)) for v in calibration['up_max_per_channel']]
        else:
            sa=[float(np.float32(v)/np.float32(127)) for v in calibration['gate_abs_max']]
            sb=[float(np.float32(v)/np.float32(32767)) for v in calibration['up_abs_max']]
        with safe_open(ROOT/'models'/folder/'model.safetensors',framework='pt',device='cpu') as weights:
            for layer in layers:
                prefix=f'model.layers.{layer}.';gamma=weights.get_tensor(prefix+'post_attention_layernorm.weight').float().numpy()
                projections=[];certs=[]
                for name in ('gate_proj','up_proj'):
                    matrix=weights.get_tensor(prefix+'mlp.'+name+'.weight').float().numpy()*gamma[None,:]
                    q,f=quantize16(matrix);upper,cert=norm_certificate(q,f);projections.append(upper);certs.append(cert)
                down=weights.get_tensor(prefix+'mlp.down_proj.weight').float().numpy();n,d=certs[0]['shape'];nr=root_upper(n);dr=root_upper(d)
                for fg in (20,22):
                    spec=build(sa[layer],8,fg,32);gain,constant=gate_linear_bound(spec)
                    norm_a=dr*projections[0]
                    norm_g=min(nr*spec['absolute_bound'],gain*Fraction(256,1)/Fraction.from_float(sa[layer])*norm_a+nr*(Fraction(gain,2)+constant))
                    norm_b=min(nr*32767,dr*projections[1]/Fraction.from_float(sb[layer])+nr*Fraction(257,512))
                    for fw in (12,14,16):
                        wq=np.rint(down.astype(np.float64)*(1<<fw)).astype(np.int64)
                        if int(wq.min())<-32768 or int(wq.max())>32767:
                            rows.append(dict(model=model,layer=layer,gate_fraction=fg,down_fraction=fw,weight16_fits=False));continue
                        maximum=int(np.abs(wq).max());l1=int(np.abs(wq).sum(1).max());joint=maximum*norm_g*norm_b;box=l1*spec['absolute_bound']*32768
                        bound=(joint.numerator+joint.denominator-1)//joint.denominator
                        row=dict(model=model,layer=layer,gate_fraction=fg,down_fraction=fw,weight16_fits=True,
                            joint_integer_accumulator_bound=bound,independent_box_integer_bound=box,joint_signed63_safe=bound<(1<<62),box_signed63_safe=box<(1<<62),
                            box_over_joint_ratio=float(Fraction(box,1)/joint),gate_gain=gain,gate_additive_bound=constant,
                            projection_norm_certificates=certs)
                        rows.append(row);print(model,layer,'Fg',fg,'Fw',fw,'boxbits',box.bit_length(),'jointbits',bound.bit_length(),'ratio',float(Fraction(box,1)/joint),flush=True)
    (ROOT/'results/joint_swiglu_norm_screen.json').write_text(json.dumps(dict(records=rows,
        status='Necessary constructive range screen, notimplementedfullFFN',
        assumptions='Both projections come from the same z with||z||²<=hidden. Gamma foldedinto signed16 quantizedprojectionweights. Exactreal/integer targetprojections, Araw nearestfixed8, Braw nearestfixed8 thenhalfupclip16. Doesnotapplyto arbitraryindependent gateinputshares.',
        limits='Selected3layers permodel. Gram/spectral upperbounds are exactrational certificates for chosenquantizedprojectionmatrices; not claimedfor FP32 nonlinear forwardor privateRMSNorm implementation. No quality/timing/newprimitive claim.'),indent=2)+'\n')


if __name__=='__main__':main()
