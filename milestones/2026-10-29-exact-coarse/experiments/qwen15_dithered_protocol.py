"""Second-model full finite-block checks with its certified F20 precision."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from dithered_silu import build
from public_spline_contraction_protocol import run

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    stem='qwen15_dither_quality_dither_b8_s20261029_wiki4';quality=json.loads((ROOT/'results'/f'{stem}.json').read_text())
    capture=ROOT/'results'/f'{stem}_capture.npz';rows=[];variants=('selector','basis','weighted')
    with np.load(capture) as cap,np.load(ROOT/'results/qwen15_down_int12_selected.npz') as weights:
        for layer in ((14,) if args.quick else (0,14,27)):
            for repeat in range(1 if args.quick else 3):
                modes=('exact32','expanded33','compact25') if args.quick else ('exact32','compact25')
                order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
                for mode in order:
                    source=quality['specifications'][layer]
                    spec=build(source['sa'],8,quality['fraction_bits'],32 if mode=='exact32' else 33)
                    spec.update(use_dither_fss=mode!='exact32',full_classifier_reference=mode=='expanded33',promised_input_bits=32)
                    vorder=variants[repeat:]+variants[:repeat]
                    for variant in vorder:
                        row=run(cap[f'layer_{layer}_a_fixed32'],cap[f'layer_{layer}_b_fixed32'].view(np.uint64),weights[f'layer_{layer}'].astype(np.int64),
                            spec,variant,rtt=0,start_method='spawn',backend='cuda' if args.cuda else 'cpu')
                        row.update(layer=layer,repeat=repeat,mode=mode,mode_order=list(order),variant_order=list(vorder));rows.append(row)
                print('layer',layer,'repeat',repeat,'passed',flush=True)
    label='cuda' if args.cuda else 'cpu';label+='_quick' if args.quick else ''
    out=dict(results=rows,all_passed=True,model_revision=quality['model_revision'],fraction_bits=quality['fraction_bits'],
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        limits='Second modelfinite8960x1536 gate/down block withprivateBround/clip andfinalshift32. Fresh masks, same inputs forcostcomparisons; actualrho checked byexactCPUintegers. Not fullFFN,upstreambaseline,modelquality orsecurity proof.')
    (ROOT/'results'/f'qwen15_dithered_protocol_{label}.json').write_text(json.dumps(out,indent=2)+'\n')
    for mode in modes:
        for variant in variants:
            subset=[v for v in rows if v['mode']==mode and v['variant']==variant]
            print(mode,variant,'maxparty medianms',float(np.median([max(p['total_ms'] for p in v['parties']) for v in subset])),flush=True)


if __name__=='__main__':main()
