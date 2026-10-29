"""First full CUDA/socket block checks before larger benchmark runs."""
import json
import numpy as np
from fused_gated_lut import ROOT
from public_silu_spline import build
from public_spline_contraction_protocol import run


def main():
    d=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text());rows=[];layer=12
    with np.load(ROOT/'results/qwen_public_spline_f22_capture.npz') as data,np.load(ROOT/'results/qwen_down_int12_selected.npz') as w:
        for variant in ('selector','basis','weighted','wide'):
            row=run(data[f'layer_{layer}_a_fixed32'],data[f'layer_{layer}_b_fixed32'].view(np.uint64),w[f'layer_{layer}'].astype(np.int64),
                build(d['specifications'][layer]['sa'],22),variant,gpu_output=data[f'layer_{layer}_contraction_output'],start_method='spawn',backend='cuda')
            rows.append(row);print(variant,row['parties'][0]['total_ms'],'passed',flush=True)
    (ROOT/'results/qwen_public_spline_cuda_first_checks.json').write_text(json.dumps(dict(results=rows,all_passed=True),indent=2)+'\n')


if __name__=='__main__':main()
