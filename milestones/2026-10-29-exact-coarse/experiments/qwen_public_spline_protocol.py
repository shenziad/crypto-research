"""Real matched private spline/B-quantization/down-projection block runs."""
import argparse
import hashlib
import json
import numpy as np
from public_silu_spline import build
from public_spline_contraction_protocol import ROOT,run
from dcf_wide_native import LIB


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--norm-views',action='store_true')
    parser.add_argument('--spawn',action='store_true')
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--frontier',action='store_true');args=parser.parse_args()
    if args.cuda:args.spawn=True
    quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text())
    norms=json.loads((ROOT/'results/qwen_norm_projection_bounds.json').read_text()) if args.norm_views else None
    capture=ROOT/'results/qwen_public_spline_f22_capture.npz';weights=ROOT/'results/qwen_down_int12_selected.npz';rows=[]
    with np.load(capture) as data,np.load(weights) as matrices:
        for layer in (0,12,23):
            source=quality['specifications'][layer]
            view=next(v['bounds']['2']['signed_input_bits'] for v in norms['layers'] if v['layer']==layer and v['projection']=='gate_proj') if norms else 32
            spec=build(source['sa'],22,view)
            a=data[f'layer_{layer}_a_fixed32'];b=data[f'layer_{layer}_b_fixed32'].view(np.uint64)
            w=matrices[f'layer_{layer}'].astype(np.int64);gpu=data[f'layer_{layer}_contraction_output']
            variants=('basis','wide','selector','weighted') if args.frontier else ('basis','wide')
            orders=[variants[i:]+variants[:i] for i in range(len(variants))] if args.frontier else [('basis','wide'),('wide','basis')]*3
            for rtt in (0,50):
                for repeat,order in enumerate(orders):
                    for variant in order:
                        r=run(a,b,w,spec,variant,rtt=rtt,gpu_output=gpu,start_method='spawn' if args.spawn else 'fork',backend='cuda' if args.cuda else 'cpu');r.update(layer=layer,repeat=repeat,order=list(order));rows.append(r)
                print('finished layer',layer,'classifier',view,'emulated RTT',rtt,flush=True)
    out=dict(results=rows,all_passed=True,capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        classifier_contract='Conditional RMSNorm budget2 range, checked on captured values; finite RMSNorm proof pending.' if norms else 'Signed32 values carried as ring63 shares; B is full signed32 fixed8. Same interface for both algorithms.',
        scheduling='All variants use five exchanges. A opening is batched with B rounding. Basis/wide open masked F with B clipping. Selector opens masked slope with B clipping and masked F with masked B. Weighted opens only B clipping then masked B, using correlated F and -rB*F shares.',
        backend='cuda' if args.cuda else 'cpu',
        limits='Local trusted-parent reference harness, not deployment. Workers receive own input/key shares; use process_context to distinguish spawn/fork. CPU uses optimized endpoint traversal; CUDA uses per-query threads, one shared physical GPU and full online public compilation/host-device/socket costs. Public operators/scales and fixed-width types. CUDA setup/upload/warmup and model initialization counted separately. Emulated delay is not WAN; dealer network and upstream private projections/input encoding excluded. No upstreamGPU, novelty or full secureLLM speedup claim.')
    if args.cuda:
        from dcf_cuda import PROTOCOL_LIB,LIB as GPU_LIB
        out['gpu_native_sha256']={v.name:hashlib.sha256(v.read_bytes()).hexdigest() for v in (PROTOCOL_LIB,GPU_LIB)}
    suffix=('_frontier' if args.frontier else '')+('_norm_views' if args.norm_views else '')+('_spawn' if args.spawn else '')+('_cuda' if args.cuda else '')
    (ROOT/'results'/f'qwen_public_spline_protocol{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    for rtt in (0,50):
        for variant in variants:
            r=[v for v in rows if v['variant']==variant and v['emulated_rtt_ms']==rtt]
            print(rtt,variant,'median party0 ms',np.median([v['parties'][0]['total_ms'] for v in r]),
                'online bytes',r[0]['online_sent_bytes_both_parties'],'keys',r[0]['total_key_array_bytes_both_parties'],
                'median dealer ms',np.median([v['dealer_ms'] for v in r]),flush=True)


if __name__=='__main__':main()
