"""Weighted spectral-tail range certificate for a future common-input FFN.

For weights w>=0, lambda_max(M.T diag(w) M) <=
min_theta[theta*R_M + sum_j (w_j-theta)_+ ||M_j||²],
where R_M certifies lambda_max(M.T M). Exact integer tail computation.
"""
import json
import hashlib
from fractions import Fraction
import numpy as np
from safetensors import safe_open
from joint_swiglu_norm_screen import ROOT,quantize16,norm_certificate,root_upper,ceil_sqrt,gate_linear_bound
from dithered_silu import build


def tail_bound(weights,row_norm_squared,global_radius):
    w=np.asarray(weights,dtype=np.int64);t=np.asarray(row_norm_squared,dtype=np.int64);maximum=int(w.max())
    assert maximum*max(int(t.sum()),int(global_radius))<(1<<62)
    order=np.argsort(w)[::-1];s=w[order];r=t[order]
    prefix_r=np.cumsum(r,dtype=np.int64);prefix_wr=np.cumsum(s*r,dtype=np.int64)
    candidates=s*global_radius+prefix_wr-s*prefix_r
    # theta=0 gives weighted trace, also a valid spectral upper bound.
    return min(int(candidates.min()),int(prefix_wr[-1]))


def main():
    records=[];matrices=[]
    for model,folder,layers in [('qwen05','Qwen2.5-0.5B',(0,12,23)),('qwen15','Qwen2.5-1.5B',(0,14,27))]:
        if model=='qwen05':
            source=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());ranges=json.loads((ROOT/'results/qwen_gate_ranges.json').read_text())
            sa=[s['sa'] for s in source['specifications']];sb=[float(np.float32(max(v))/np.float32(32767)) for v in ranges['up_max_per_channel']]
        else:
            ranges=json.loads((ROOT/'results/qwen15_gate_ranges.json').read_text());sa=[float(np.float32(v)/np.float32(127)) for v in ranges['gate_abs_max']];sb=[float(np.float32(v)/np.float32(32767)) for v in ranges['up_abs_max']]
        with safe_open(ROOT/'models'/folder/'model.safetensors',framework='pt',device='cpu') as file:
            for layer in layers:
                prefix=f'model.layers.{layer}.';gamma=file.get_tensor(prefix+'post_attention_layernorm.weight').float().numpy();qs=[];certs=[]
                for name in ('gate_proj','up_proj'):
                    eff=file.get_tensor(prefix+'mlp.'+name+'.weight').float().numpy()*gamma[None,:];q,f=quantize16(eff);_,cert=norm_certificate(q,f);qs.append(q);certs.append(cert)
                n,d=qs[0].shape;dr=root_upper(d);norms=[(q.astype(np.int64)**2).sum(1) for q in qs]
                matrices.append(dict(model=model,layer=layer,projection_certificates=certs))
                down=file.get_tensor(prefix+'mlp.down_proj.weight').float().numpy()
                for fw in (12,14,16):
                    w=np.rint(down.astype(np.float64)*(1<<fw)).astype(np.int64)
                    if int(w.min())<-32768 or int(w.max())>32767:
                        records.append(dict(model=model,layer=layer,down_fraction=fw,weight16_fits=False));continue
                    wa=np.abs(w);spectral=[]
                    for i in range(d):
                        radii=[tail_bound(wa[i],norms[k],certs[k]['gershgorin_gram_row_absolute_sum']) for k in (0,1)]
                        spectral.append([Fraction(ceil_sqrt(radii[k]),1<<certs[k]['weight_fraction_bits']) for k in (0,1)])
                    for fg in (20,22):
                        spec=build(sa[layer],8,fg,32);gain,constant=gate_linear_bound(spec);bounds=[];boxes=[]
                        for i in range(d):
                            l1=int(wa[i].sum());sr=root_upper(l1);ga=gain*Fraction(256,1)/Fraction.from_float(sa[layer])*dr*spectral[i][0]+sr*(Fraction(gain,2)+constant)
                            ga=min(ga,sr*spec['absolute_bound']);bu=dr*spectral[i][1]/Fraction.from_float(sb[layer])+sr*Fraction(257,512);bu=min(bu,sr*32767)
                            candidate=ga*bu;box=l1*spec['absolute_bound']*32768;bound=min(box,(candidate.numerator+candidate.denominator-1)//candidate.denominator)
                            bounds.append(bound);boxes.append(box)
                        maximum=max(bounds);old=max(boxes);rows_improved=sum(a<b for a,b in zip(bounds,boxes))
                        rec=dict(model=model,layer=layer,gate_fraction=fg,down_fraction=fw,weight16_fits=True,all_down_rows=d,
                            weighted_integer_accumulator_bound=maximum,independent_box_integer_bound=old,weighted_signed63_safe=maximum<(1<<62),box_signed63_safe=old<(1<<62),
                            box_over_weighted_ratio=old/maximum,strictly_improved_rows=rows_improved)
                        records.append(rec);print(model,layer,'Fg',fg,'Fw',fw,'boxbits',old.bit_length(),'weightedbits',maximum.bit_length(),'ratio',old/maximum,flush=True)
    out=dict(records=records,projection_certificates=matrices,
        assumptions='Common RMSunitinput z with||z||²<=hidden; actual signed16 quantizedgammafolded projections; exactnearestAraw/Braw encodings thenhalfupclipB. Doesnotapplyto independently suppliedA/B orwithouta finiteRMSinputnorm contract.',
        limits='Necessary mathematical range screen on3layers permodel, alloutputrows. No privatefullFFN,FP32-globalcertificate,quality,timing ornovelty conclusion. Takingmin witholdbox explicitly prevents reportingalooser candidate asanimprovement.')
    (ROOT/'results/weighted_swiglu_norm_screen.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
