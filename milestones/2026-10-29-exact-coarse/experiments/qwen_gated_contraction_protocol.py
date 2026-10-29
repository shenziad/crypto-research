"""Matched real finite Qwen gate and private down-projection, local harness."""
import hashlib
import json
import numpy as np
from gated_contraction_protocol import ROOT,run
from dpf_aes_native import LIB


def main():
    modes=('beaver56','mixed16x56','beaver32_lift','mixed16x16_lift','fused56')
    orders=[modes[i:]+modes[:i] for i in range(len(modes))];rows=[]
    capture=ROOT/'results/qwen_gate_fixed8_input_late_capture.npz'
    weights=ROOT/'results/qwen_down_int12_selected.npz'
    with np.load(capture) as data,np.load(weights) as matrices:
        for layer in (0,12,23):
            prefix=f'layer_layer_{layer}';w=matrices[f'layer_{layer}'].astype(np.int64)
            a=data[prefix+'_a_fixed32'].view(np.uint64);b=data[prefix+'_b_fixed32'].view(np.uint64)
            table=data[prefix+'_lookup']
            for rtt in (0,50):
                for repetition,order in enumerate(orders):
                    for mode in order:
                        record=run(a,b,w,table,mode,rtt=rtt);record.update(layer=layer,repetition=repetition,order=list(order));rows.append(record)
                print('finished layer',layer,'emulated RTT',rtt,flush=True)
    result=dict(results=rows,order_design='Five cyclic orders, each mode appears once in each timing position; not all120 permutations.',capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        weight_file_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        functionality='All modes: private normalized fixed8 ring32 inputs -> half-up clip -> integer gated product -> private int16 weight contraction -> exact signed64 floor /2^20',
        limits='Local trusted parent/fork, not deployment. Up/gate input matrix projections and conversion to normalized fixed8 remain outside. Model mask initialization61MB per layer counted separately. Fresh triples every measurement, no amortized preprocessing speed claim. Emulated50ms sleeps are not actual WAN. Standard FSS/LLAMA/Beaver primitives; no confirmed independent novelty.')
    (ROOT/'results/qwen_gated_contraction_protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    for rtt in (0,50):
        for mode in modes:
            r=[v for v in rows if v['mode']==mode and v['emulated_rtt_ms']==rtt]
            print(rtt,mode,'median_ms',np.median([v['parties'][0]['total_ms'] for v in r]),
                'online_bytes',r[0]['online_sent_bytes_both_parties'],'material',r[0]['key_array_bytes_both_parties'],
                'model_setup_bytes',r[0]['model_initialization_sent_bytes_both_parties'],flush=True)


if __name__=='__main__':main()
