"""GPU numerical gate experiment; projections remain original floating-point.

Quantizes all SwiGLU gate inputs to 8 bits, up inputs to 16 bits and optionally
applies the operator's unbiased 8-bit product rounding. NOT full secure inference.
Public ranges calibrated on four disjoint WikiText windows, tested on four others.
"""
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/Qwen2.5-0.5B'
DATA=ROOT/'data/wikitext2_test.txt'
CAP=ROOT/'results/qwen_gate_capture.npz'


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    text=DATA.read_text();tokens=tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    length=129;calibration=[0,512,1024,1536];test=[4096,4608,5120,5632]
    assert len(tokens)>max(test)+length
    mlps=[layer.mlp for layer in model.model.layers]
    ranges_a=[torch.zeros(model.config.intermediate_size,device='cuda') for _ in mlps]
    ranges_b=[torch.zeros_like(ranges_a[0]) for _ in mlps]
    hooks=[]
    def capture(values,i):
        def hook(module,args,output):
            values[i]=torch.maximum(values[i],output.detach().abs().amax(dim=(0,1)))
        return hook
    for i,m in enumerate(mlps):
        hooks.extend([m.gate_proj.register_forward_hook(capture(ranges_a,i)),m.up_proj.register_forward_hook(capture(ranges_b,i))])
    with torch.inference_mode():
        for offset in calibration:
            model(tokens[offset:offset+length][None].to('cuda'),use_cache=False)
    for hook in hooks:hook.remove()
    amax=torch.stack(ranges_a).clamp_min(0.125)*1.1
    bmax=torch.stack(ranges_b).clamp_min(0.125)*1.1
    ranges={'gate_max_per_channel':amax.cpu().tolist(),'up_max_per_channel':bmax.cpu().tolist(),
            'calibration_offsets':calibration,'test_offsets':test,'length':length,'range_factor':1.1,
            'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest()}
    (ROOT/'results/qwen_gate_ranges.json').write_text(json.dumps(ranges,indent=2)+'\n')
    originals=[m.forward for m in mlps];rows=[];captured={};captures_done=False
    for offset in test:
        ids=tokens[offset:offset+length][None].to('cuda')
        with torch.inference_mode():
            baseline=model(ids,use_cache=False).logits[0,:-1].float()
        logp=baseline.log_softmax(-1);prob=logp.exp();target=ids[0,1:]
        base_nll=float((-logp[torch.arange(length-1,device='cuda'),target]).mean())
        for layout,rounded,seed in [('layer',False,0),('layer',True,292001),('layer',True,292002),('channel',False,0),('channel',True,292001),('channel',True,292002)]:
            stats=[{'gate_clipped':0,'up_clipped':0,'elements':0,'max_abs_raw_product':0} for _ in mlps]
            generator=torch.Generator(device='cuda');generator.manual_seed(seed)
            def make_forward(i):
                sa=(amax[i].max() if layout=='layer' else amax[i])/127
                sb=(bmax[i].max() if layout=='layer' else bmax[i])/32767
                def forward(self,x):
                    a=self.gate_proj(x);b=self.up_proj(x)
                    ai=torch.round(a/sa).clamp(-127,127).to(torch.int64)
                    bi=torch.round(b/sb).clamp(-32767,32767).to(torch.int64)
                    ga=torch.nn.functional.silu(ai.float()*sa)
                    gq=torch.round(ga*256).to(torch.int64)
                    raw=gq*bi
                    stats[i]['gate_clipped']+=int((a.abs()>sa*127).sum())
                    stats[i]['up_clipped']+=int((b.abs()>sb*32767).sum())
                    stats[i]['elements']+=a.numel()
                    stats[i]['max_abs_raw_product']=max(stats[i]['max_abs_raw_product'],int(raw.abs().max()))
                    if rounded:
                        floor=torch.div(raw,256,rounding_mode='floor')
                        r=raw-floor*256
                        coin=torch.rand(r.shape,generator=generator,device='cuda')<r.float()/256
                        product=(floor+coin.to(torch.int64)).float()*sb
                    else:product=raw.float()*(sb/256)
                    if not captures_done and not rounded and i in (0,12,23):
                        # First real test token at these layers. Inputs at layer
                        # >0 already include preceding approximate gates.
                        signed=np.arange(256,dtype=np.int64);signed=(signed^128)-128
                        table_a=torch.tensor(signed,device='cuda',dtype=torch.float32)[:,None]*sa
                        table=torch.round(torch.nn.functional.silu(table_a)*256).to(torch.int64).T
                        prefix=f'{layout}_layer_{i}'
                        captured[f'{prefix}_a_index']=np.remainder(ai[0,0].cpu().numpy(),256).astype(np.uint8)
                        captured[f'{prefix}_b_integer']=bi[0,0].cpu().numpy()
                        captured[f'{prefix}_lookup']=table[0].cpu().numpy() if sa.ndim==0 else table.cpu().numpy()
                        captured[f'{prefix}_up_scale']=sb.detach().cpu().numpy()
                        captured[f'{prefix}_a_real']=a[0,0].cpu().numpy()
                        captured[f'{prefix}_b_real']=b[0,0].cpu().numpy()
                    return self.down_proj(product)
                return forward
            for i,m in enumerate(mlps):m.forward=types.MethodType(make_forward(i),m)
            with torch.inference_mode():
                candidate=model(ids,use_cache=False).logits[0,:-1].float()
            for m,original in zip(mlps,originals):m.forward=original
            logq=candidate.log_softmax(-1)
            row={'test_offset':offset,'layout':layout,'rounded':rounded,'round_seed':seed,
                 'nll_baseline':base_nll,'nll_candidate':float((-logq[torch.arange(length-1,device='cuda'),target]).mean()),
                 'mean_kl':float((prob*(logp-logq)).sum(-1).mean()),
                 'top1_agreement':float((baseline.argmax(-1)==candidate.argmax(-1)).float().mean()),
                 'layer_stats':stats}
            rows.append(row)
            print(offset,layout,rounded,seed,'NLL delta',row['nll_candidate']-base_nll,'KL',row['mean_kl'],flush=True)
        captures_done=True
    np.savez_compressed(CAP,**captured)
    result={'model':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
            'model_sha256':'88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342',
            'hardware':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'float32','tf32':False,
            'calibration_offsets':calibration,'test_offsets':test,'context_tokens':length,'records':rows,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'capture':str(CAP),
            'limits':'Numerical gate proxy only. All 24 MLP gates changed; all matrix projections, attention and norms remain original floating point, no MPC or protocol time. Four calibration/test windows from same WikiText2 test corpus, disjoint offsets; not whole-corpus PPL. Input quantization/privacy and ring lifting/projection costs excluded. Captured deeper-layer inputs already reflect preceding approximate gates. Stochastic round uses torch simulation, not real dealer.'}
    (ROOT/'results/qwen_gated_quality.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
