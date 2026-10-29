"""Matched original exact32, shared-correction exact24 and randomized compact25."""
import argparse
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from dithered_silu import build
from public_spline_contraction_protocol import run as direct_run
from exact_coarse_contraction_protocol import run as corrected_run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=('qwen05','qwen15'),default='qwen05');parser.add_argument('--cuda',action='store_true');parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    if args.model=='qwen05':
        quality='qwen_public_spline_quality_f22_alignedb8.json';capture='qwen_public_spline_f22_alignedb8_capture.npz';weights='qwen_down_int12_selected.npz';layers=(0,12,23)
    else:
        quality='qwen15_dither_quality_aligned_b8_s20261029_wiki4.json';capture='qwen15_dither_quality_aligned_b8_s20261029_wiki4_capture.npz';weights='qwen15_down_int12_selected.npz';layers=(0,14,27)
    source=json.loads((ROOT/'results'/quality).read_text());variants=('basis','weighted');modes=('exact32','corrected24','compact25');rows=[]
    with np.load(ROOT/'results'/capture) as cap,np.load(ROOT/'results'/weights) as wm:
        for layer in ((layers[1],) if args.quick else layers):
            for rtt in ((0,) if args.quick else (0,50)):
                for repeat in range(1 if args.quick else 3):
                    for mode in modes[repeat:]+modes[:repeat]:
                        original=source['specifications'][layer];spec=build(original['sa'],8,original['fraction_bits'],33 if mode=='compact25' else 32)
                        spec.update(use_dither_fss=mode!='exact32',promised_input_bits=32)
                        for variant in variants[repeat%2:]+variants[:repeat%2]:
                            call=corrected_run if mode=='corrected24' else direct_run
                            kwargs=dict(rtt=rtt,gpu_output=None if mode=='compact25' else cap[f'layer_{layer}_contraction_output'],backend='cuda' if args.cuda else 'cpu')
                            if mode!='corrected24':kwargs['start_method']='spawn'
                            row=call(cap[f'layer_{layer}_a_fixed32'],cap[f'layer_{layer}_b_fixed32'].view(np.uint64),wm[f'layer_{layer}'].astype(np.int64),spec,variant,**kwargs)
                            row.update(mode=mode,layer=layer,repeat=repeat);rows.append(row)
                    print(layer,rtt,repeat,'passed',flush=True)
    suffix=f'{args.model}_'+('cuda' if args.cuda else 'cpu')+('_quick' if args.quick else '')
    out=dict(results=rows,all_passed=True,source_sha256={name:hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest() for name in (quality,capture,weights)},
        limits='Exact32 andcorrected24 same deterministic aligned finite function, checked against GPU capture and exact CPU intermediates. Compact25 changes the function withpublicrho. Five exchanges each; corrected24 sends an extra4 bytes perparty perinput. Trusted parent, one sharedGPU, no upstreambaseline orsecurity/novelty claim.')
    (ROOT/'results'/f'qwen_exact_coarse_compare_{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    for mode in modes:
        for variant in variants:
            subset=[r for r in rows if r['mode']==mode and r['variant']==variant and r['emulated_rtt_ms']==0]
            print(mode,variant,'maxparty medianms',float(np.median([max(p['total_ms'] for p in r['parties']) for r in subset])),
                'onlineB',subset[0]['online_sent_bytes_both_parties'],'keyB',subset[0]['total_key_array_bytes_both_parties'],flush=True)


if __name__=='__main__':main()
