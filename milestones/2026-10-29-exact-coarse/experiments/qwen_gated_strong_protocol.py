"""Real Qwen finite gates, three matched baselines through signed64 output."""
import hashlib
import argparse
import itertools
import json
import numpy as np
from gated_lut_strong_baselines import run,ROOT
from dpf_aes_native import LIB


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--postprocess',choices=('faithful','legacy_stochastic'),default='faithful')
    args=parser.parse_args()
    capture=ROOT/'results/qwen_gate_capture.npz'
    certificate=json.loads((ROOT/'results/qwen_gate_integer_certificate.json').read_text())
    tables=np.load(ROOT/'results/qwen_layer_lut_tables.npz')['tables']
    assert certificate['all_layers_passed']
    modes=('beaver32','mixed16','fused32');orders=list(itertools.permutations(modes));rows=[]
    with np.load(capture) as data:
        for layer in (0,12,23):
            prefix=f'layer_layer_{layer}'
            a=data[prefix+'_a_index'];b=data[prefix+'_b_integer'].view(np.uint64)
            table=data[prefix+'_lookup'].view(np.uint64)
            assert np.array_equal(table.view(np.int64),tables[layer])
            assert hashlib.sha256(tables[layer].astype('<i8').tobytes()).hexdigest()==certificate['layers'][layer]['table_sha256']
            for rtt in (0,50):
                for repetition,order in enumerate(orders):
                    for mode in order:
                        record=run(a,b,mode,table_values=table,rtt_ms=rtt,postprocess=args.postprocess)
                        record.update(layer=layer,repetition=repetition,order=list(order));rows.append(record)
                print('finished real layer',layer,'emulated RTT',rtt,flush=True)
    out={'description':'Same captured Qwen values and per-layer finite table; a8/ring32(b16-value) input shares, signed64 rounded output shares',
         'postprocess':args.postprocess,
         'results':rows,'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),
         'native_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
         'integer_range_certificate':'qwen_gate_integer_certificate.json',
         'limits':'Six complete algorithm orders per layer/delay, three real captured layer vectors. Trusted parent/fork harness, local emulated50ms exchanges, not WAN. Model projections, input quantization, dealer network and output reconstruction excluded. LLAMA mixed multiplication and sign extension are prior constructions.'}
    filename='qwen_gated_strong_faithful.json' if args.postprocess=='faithful' else 'qwen_gated_strong_legacy_stochastic.json'
    (ROOT/'results'/filename).write_text(json.dumps(out,indent=2)+'\n')
    for rtt in (0,50):
        for mode in modes:
            r=[v for v in rows if v['mode']==mode and v['emulated_rtt_ms_per_logical_exchange']==rtt]
            print(rtt,mode,'median_total_ms',np.median([v['parties'][0]['total_ms'] for v in r]),
                  'online_bytes',r[0]['online_sent_bytes_both_parties'],'total_material_bytes',r[0]['total_key_array_bytes_both_parties'])


if __name__=='__main__':main()
