"""Enumerate all layerwise finite SiLU LUTs and certify ring32/24 ranges.

Certificate binds the finite integer tables, not continuous SiLU accuracy or
untrusted range enforcement. Input b must be a signed16 integer by functionality.
"""
import hashlib
import json

import numpy as np
import torch

from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'results/qwen_gate_ranges.json'
    ranges=json.loads(path.read_text())
    assert torch.cuda.is_available()
    amax=torch.tensor(ranges['gate_max_per_channel'],device='cuda',dtype=torch.float32)
    sa=amax.max(dim=1).values/127
    signed=torch.arange(256,device='cuda',dtype=torch.int64);signed=(signed^128)-128
    tables=torch.round(torch.nn.functional.silu(sa[:,None]*signed.float()[None,:])*256).to(torch.int64).cpu().numpy()
    with np.load(ROOT/'results/qwen_gate_capture.npz') as capture:
        for layer in (0,12,23):assert np.array_equal(tables[layer],capture[f'layer_layer_{layer}_lookup'])
    records=[]
    for layer,table in enumerate(tables):
        m=max(abs(int(v)) for v in table)
        raw_bound=32768*m;rounded_bound=(raw_bound+255)//256
        assert raw_bound<=(1<<31)-1 and rounded_bound<=(1<<23)-1
        records.append({'layer':layer,'table_integer_min':int(table.min()),'table_integer_max':int(table.max()),
                        'max_abs_table_integer':m,'b_abs_bound':32768,'raw_abs_bound':raw_bound,
                        'stochastically_rounded_abs_bound':rounded_bound,'raw_signed_ring_bits':32,'rounded_signed_ring_bits':24,
                        'table_sha256':hashlib.sha256(table.astype('<i8').tobytes()).hexdigest()})
    np.savez_compressed(ROOT/'results/qwen_layer_lut_tables.npz',tables=tables)
    result={'model':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
            'range_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'finite_table_generation':'float32 CUDA torch SiLU at all256 encoded inputs, rounded output*256; resulting integer table is the specified functionality',
            'layers':records,'all_layers_passed':True,'max_raw_abs_bound':max(r['raw_abs_bound'] for r in records),
            'max_rounded_abs_bound':max(r['stochastically_rounded_abs_bound'] for r in records),
            'limits':'Exact Python integer inequalities over serialized finite integer tables and signed16 b. Does not prove closeness to continuous SiLU, input quantization/enforcement privacy, final model error or lift correctness.'}
    (ROOT/'results/qwen_gate_integer_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2))


if __name__=='__main__':main()
