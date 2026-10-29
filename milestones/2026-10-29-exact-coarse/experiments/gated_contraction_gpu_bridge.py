"""Cross-check GPU exact integer contraction with CPU and Python big ints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    capture=ROOT/'results/qwen_gate_fixed8_input_late_capture.npz'
    weights=ROOT/'results/qwen_down_int12_selected.npz'
    cert=json.loads((ROOT/'results/qwen_gated_quality_fixed8_input_late.json').read_text())['contraction_integer_certificates']
    with np.load(capture) as data,np.load(weights) as matrices:
        for layer in (0,12,23):
            prefix=f'layer_layer_{layer}'
            a=np.clip((data[prefix+'_a_fixed32']+128)//256,-127,127)
            b=np.clip((data[prefix+'_b_fixed32']+128)//256,-32767,32767)
            assert np.array_equal((a&255).astype(np.uint8),data[prefix+'_a_index'])
            assert np.array_equal(b,data[prefix+'_b_integer'])
            raw=data[prefix+'_lookup'][a&255]*b;w=matrices[f'layer_{layer}'].astype(np.int64)
            expected=w@raw
            bound=cert[layer]['absolute_accumulator_bound'];assert bound<(1<<53)
            gpu=(torch.tensor(w,device='cuda',dtype=torch.float64)@torch.tensor(raw,device='cuda',dtype=torch.float64)).to(torch.int64).cpu().numpy()
            assert np.array_equal(gpu,expected)
            for index in (0,17,89,len(w)-1):
                assert sum(int(x)*int(y) for x,y in zip(w[index],raw))==int(expected[index])
            output=expected//(1<<20)
            rows.append(dict(layer=layer,all_896_accumulators_equal=True,python_bigint_rows=4,
                output_sha256=hashlib.sha256(output.astype('<i8').tobytes()).hexdigest(),
                output_max_abs=int(np.abs(output).max()),all_domain_integer_bound=bound))
    result=dict(layers=rows,all_passed=True,hardware=torch.cuda.get_device_name(),
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        limits='Numerical bridge for selected real vectors, not private GPU inference. Two-party CPU protocol independently checks the same finite integer contraction.')
    (ROOT/'results/gated_contraction_gpu_bridge.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
