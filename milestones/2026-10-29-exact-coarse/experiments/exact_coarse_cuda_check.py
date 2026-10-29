"""Individual-party low-correction and raw polynomial shares on CPU vs CUDA."""
import json
import numpy as np
import torch
from fused_gated_lut import ROOT
from dithered_silu import build,evaluate as plain
from exact_coarse_spline import prepare,masked_classifier,raw,evaluate
from exact_coarse_cuda import compile_public
from public_spline_cuda import prepare_runtime,raw as gpu_raw,evaluate as gpu_evaluate
from public_spline_cuda_protocol import upload
from dcf_cuda import select


def main():
    rows=[];mask=np.uint64((1<<63)-1)
    for model,layer,quality,capture in [('qwen05',12,'qwen_public_spline_quality_f22_alignedb8.json','qwen_public_spline_f22_alignedb8_capture.npz'),
        ('qwen15',14,'qwen15_dither_quality_aligned_b8_s20261029_wiki4.json','qwen15_dither_quality_aligned_b8_s20261029_wiki4_capture.npz')]:
        source=json.loads((ROOT/'results'/quality).read_text());original=source['specifications'][layer];spec=build(original['sa'],8,original['fraction_bits'],32);spec['use_dither_fss']=True
        with np.load(ROOT/'results'/capture) as cap:a=cap[f'layer_{layer}_a_fixed32']
        for variant in ('basis','weighted'):
            keys,alpha,gamma=prepare(len(a),spec,63,variant);h=(a.view(np.uint64)+alpha)&mask;vm=np.uint64((1<<24)-1)
            qs=[masked_classifier(keys[p],h,spec,p) for p in (0,1)];t=(qs[0]+qs[1])&vm;assert np.array_equal(t,(gamma+(a//256).view(np.uint64))&vm)
            outputs=[]
            for p in (0,1):
                rt=prepare_runtime({k:v for k,v in keys[p].items() if not k.startswith('classifier_')},h,spec,variant)
                ht=upload(h);tt=upload(t);compile_public(rt,ht,tt)
                low=select(upload(keys[p]['classifier_low']),(ht&255)[:,None].contiguous(),upload(np.arange(1,dtype=np.int32)),p,8,1,32,True)[:,0,0]
                q=upload(keys[p]['classifier_constant'])+low
                if p==0:q=q+((ht&((1<<32)-1))>>8)
                q=q&((1<<24)-1);assert np.array_equal(q.cpu().numpy().view(np.uint64),qs[p])
                expected_raw=raw(keys[p]['dcf'],h,t,spec,p,63,variant);got_raw=gpu_raw(rt,p)
                expected_raw=expected_raw if variant=='weighted' else expected_raw[:,0]
                assert np.array_equal(got_raw.cpu().numpy().view(np.uint64),expected_raw)
                expected=evaluate(keys[p],h,t,spec,p,63,variant);got=gpu_evaluate(rt,p).cpu().numpy().view(np.uint64)
                assert np.array_equal(got,expected);outputs.append(got)
            total=(outputs[0]+outputs[1])&mask;target=plain(spec,a).view(np.uint64)&mask
            assert np.array_equal(total[:,0] if variant=='weighted' else total,target)
            rows.append(dict(model=model,layer=layer,n=len(a),variant=variant,classifier_bits=24,body_bits=63,individual_party_raw_equal=True,individual_party_low_equal=True,passed=True));print(rows[-1],flush=True)
    (ROOT/'results/exact_coarse_cuda_check.json').write_text(json.dumps(dict(records=rows,all_passed=True,hardware=torch.cuda.get_device_name(),
        limits='Exact serialized-key functional checks against CPU; no timing or newsecurity/novelty claim.'),indent=2)+'\n')


if __name__=='__main__':main()
