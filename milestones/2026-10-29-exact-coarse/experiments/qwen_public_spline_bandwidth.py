"""Rate-delay sensitivity of three Pareto baselines; actual local socket checks."""
import hashlib
import argparse
import json
import numpy as np
from public_silu_spline import build
from public_spline_contraction_protocol import ROOT,run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text())
    rows=[];variants=('selector','basis','weighted')
    with np.load(ROOT/'results/qwen_public_spline_f22_capture.npz') as data,np.load(ROOT/'results/qwen_down_int12_selected.npz') as matrices:
        for rate in (10,100):
            for layer in (0,12,23):
                spec=build(quality['specifications'][layer]['sa'],22,32)
                for repeat in range(3):
                    order=variants[repeat:]+variants[:repeat]
                    for variant in order:
                        row=run(data[f'layer_{layer}_a_fixed32'],data[f'layer_{layer}_b_fixed32'].view(np.uint64),
                            matrices[f'layer_{layer}'].astype(np.int64),spec,variant,rtt=20,
                            gpu_output=data[f'layer_{layer}_contraction_output'],bandwidth_mbps=rate,
                            backend='cuda' if args.cuda else 'cpu',start_method='spawn' if args.cuda else 'fork')
                        row.update(layer=layer,repeat=repeat,order=list(order));rows.append(row)
                print('finished Mbps',rate,'layer',layer,flush=True)
    out=dict(results=rows,all_passed=True,model='Full duplex payload delay: each party sleeps 8*sent_bytes/(Mbps*1e6) per online exchange, plus20ms/exchange. UNIX sockets transfer before delay. No TCP, queueing, packet loss, real WAN or dealer bandwidth emulation.',
        fairness='Same data/function, three cyclic orders; model initialization bytes recorded separately and not subject to payload delay.',backend='cuda' if args.cuda else 'cpu',
        protocol_source_sha256=hashlib.sha256((ROOT/'experiments/public_spline_contraction_protocol.py').read_bytes()).hexdigest())
    suffix='_cuda' if args.cuda else ''
    (ROOT/'results'/f'qwen_public_spline_bandwidth{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    for rate in (10,100):
        for variant in variants:
            subset=[v for v in rows if v['variant']==variant and v['payload_delay_mbps_per_direction']==rate]
            print(rate,variant,float(np.median([v['parties'][0]['total_ms'] for v in subset])),flush=True)


if __name__=='__main__':main()
