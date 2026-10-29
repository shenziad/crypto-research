"""Matched exact32 / full33 same-randomized-function / compact25 block runs."""
import argparse
import hashlib
import json
import numpy as np
from fused_gated_lut import ROOT
from dithered_silu import build
from public_spline_contraction_protocol import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    source=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());rows=[];variants=('selector','basis','weighted')
    capture=ROOT/'results/qwen_public_spline_f22_ditherb8_seed20261029_capture.npz'
    with np.load(capture) as data,np.load(ROOT/'results/qwen_down_int12_selected.npz') as w:
        for layer in ((12,) if args.quick else (0,12,23)):
            for rtt in ((0,) if args.quick else (0,50)):
                modes=('exact32','expanded33','compact25')
                for repeat in range(1 if args.quick else 3):
                    mode_order=modes[repeat:]+modes[:repeat]
                    for mode in mode_order:
                        spec=build(source['specifications'][layer]['sa'],8,input_bits=32 if mode=='exact32' else 33)
                        spec.update(use_dither_fss=mode!='exact32',full_classifier_reference=mode=='expanded33',promised_input_bits=32)
                        order=variants[repeat:]+variants[:repeat]
                        for variant in order:
                            row=run(data[f'layer_{layer}_a_fixed32'],data[f'layer_{layer}_b_fixed32'].view(np.uint64),w[f'layer_{layer}'].astype(np.int64),spec,variant,rtt=rtt,
                                start_method='spawn',backend='cuda' if args.cuda else 'cpu')
                            row.update(layer=layer,mode=mode,repeat=repeat,variant_order=list(order),mode_order=list(mode_order));rows.append(row)
                            if args.quick:print(mode,variant,row['parties'][0]['total_ms'],'passed',flush=True)
                print('finished layer',layer,'RTT',rtt,flush=True)
    suffix=('_cuda' if args.cuda else '_cpu')+('_quick' if args.quick else '')
    out=dict(results=rows,all_passed=True,capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        comparison='Exact32 uses the same aligned C0 fine function without classifier randomization. Expanded33 andcompact25 compute the SAME public-dither randomized function with fresh masks; expanded classification input shifts by B-1-rho, body input stays unchanged.',
        limits='Trusted parent reference. Both parties independently spawned; private B quantizer, gate, private down and final shift checked against exact CPU integers. Frozen numerical proxy separately evaluates full-model quality. No original GPU capture equality claimed for fresh independent dither; no novelty/upstream/fullLLM claim.')
    (ROOT/'results'/f'qwen_dithered_protocol{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    for mode in modes:
        for variant in variants:
            subset=[v for v in rows if v['mode']==mode and v['variant']==variant and v['emulated_rtt_ms']==0]
            print(mode,variant,'median max-party ms',float(np.median([max(p['total_ms'] for p in v['parties']) for v in subset])),
                'keys',subset[0]['total_key_array_bytes_both_parties'],'online',subset[0]['online_sent_bytes_both_parties'],flush=True)


if __name__=='__main__':main()
