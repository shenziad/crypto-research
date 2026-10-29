"""Exact integer continuity/boundary/accumulator certificates; finite grid screen."""
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from dithered_silu import build,evaluate


def main():
    source=ROOT/'results/qwen_public_spline_quality_f22.json';q=json.loads(source.read_text());rows=[]
    x=np.arange(-32512,32513,dtype=np.int64)
    for shift in (4,6,8):
        group=[]
        for i,s in enumerate(q['specifications']):
            spec=build(s['sa'],shift);a=x*s['sa']/256;truth=a/(1+np.exp(-a));fine=evaluate(spec,x)/(1<<22)
            l1=q['integer_certificates'][i]['maximum_weight_integer_l1'];gmax=spec['absolute_bound']
            full=l1*gmax*32768;high=l1*((gmax+16383)//16384)*32768;low=l1*16383*32768
            assert full<(1<<62) and high<(1<<53) and low<(1<<53)
            row=dict(layer=i,shift=shift,classifier_bits=33-shift,net_bits_saved_vs_exact32=shift-1,
                maximum_grid_silu_error=float(np.max(np.abs(fine-truth))),maximum_classifier_error=spec['maximum_classifier_integer_error']/(1<<22),
                maximum_expected_classifier_bias=max(v['maximum_expected_bias'] for v in spec['boundary_error_certificate'])/(1<<22),
                absolute_integer_gate_bound=gmax,maximum_private_weight_l1=l1,
                all_domain_signed32_accumulator_bound=full,split_high_bound=high,split_low_bound=low,
                signed63_safe=True,split_float64_dots_exact=True,specification=spec)
            group.append(row);rows.append(row)
        print('shift',shift,'fine maxerror',max(v['maximum_grid_silu_error'] for v in group),'classifier maxbound',max(v['maximum_classifier_error'] for v in group),flush=True)
    out=dict(records=rows,all_integer_continuity_and_range_checks_passed=True,source_quality_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limits='Classifier-vs-aligned finite function boundary bounds are algebraic on all promised signed32 inputs, using a guarded33-bit classifier. Grid SiLU differences cover calibrated clipping interval only. Model quality, optimized protocol performance and novelty not established by this certificate.')
    (ROOT/'results/dithered_silu_certificates.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
