"""Necessary feasibility screen for deferred attention normalization.

No private protocol or speed claim. Public, calibration-frozen per-position
gauges test whether projective residual/RMSNorm representation remains viable
in a small fixed-point ring. Real multihead denominator statistics are used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'models/Qwen2.5-0.5B'


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ranges=json.loads((ROOT/'results/qwen_gate_ranges.json').read_text());length=ranges['length']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);tokenizer.model_max_length=1<<30
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    captures={};handles=[]
    for i,layer in enumerate(model.model.layers):
        def hook(module,args,i=i):captures[i]=args[0].detach()
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(hook))
    texts={label:tokenizer((ROOT/'data'/name).read_text(),return_tensors='pt',add_special_tokens=False)['input_ids'][0] for label,name in [('wiki','wikitext2_test.txt'),('ptb','ptb_test.txt')]}
    assert hashlib.sha256((ROOT/'data/wikitext2_test.txt').read_bytes()).hexdigest()==ranges['data_sha256']
    def evaluate(label,offset):
        captures.clear();ids=texts[label][offset:offset+length][None].to('cuda');assert ids.shape[-1]==length
        with torch.inference_mode():out=model.model(ids,use_cache=False,output_attentions=True,return_dict=True)
        assert len(out.attentions)==len(model.model.layers)
        logs=torch.stack([-a[0].amax(-1).double().log2().sum(0) for a in out.attentions])
        assert logs.shape==(24,length) and torch.isfinite(logs).all()
        return logs,[captures[i][0].double() for i in range(24)]
    calibration=[]
    for offset in ranges['calibration_offsets']:
        logd,_=evaluate('wiki',offset);calibration.append(logd.cpu().numpy())
    gauge=np.rint(np.median(np.stack(calibration),axis=0)).astype(np.int64)
    gauge_gpu=torch.from_numpy(gauge).to('cuda');rows=[];log_captures=[]
    for label,offsets in [('wiki',list(range(8192,8192+32*512,512))),('ptb',list(range(0,32*1024,1024)))]:
        for index,offset in enumerate(offsets):
            logd,inputs=evaluate(label,offset);centered=logd-gauge_gpu;log_captures.append(dict(corpus=label,offset=offset,logd=logd.cpu().numpy()))
            for layer,x in enumerate(inputs):
                d=torch.exp2(centered[layer]);eps=model.model.layers[layer].post_attention_layernorm.variance_epsilon
                gamma=model.model.layers[layer].post_attention_layernorm.weight.double()
                target=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*gamma
                # Real identity includes eps*d^2. It is checked before any
                # fixed-point screen; omission would change the function.
                exact=x*d[:,None];identity=exact*torch.rsqrt(exact.square().mean(-1,keepdim=True)+eps*d[:,None].square())*gamma
                assert torch.allclose(identity,target,atol=1e-10,rtol=1e-10)
                for fraction in (16,24,32):
                    dq=torch.round(d*(1<<fraction));u=torch.round(x*d[:,None]*(1<<fraction))
                    denominator=u.square().mean(-1)+eps*dq.square();valid=denominator>0
                    candidate=torch.where(valid[:,None],u*torch.rsqrt(denominator)[:,None],torch.zeros_like(u))*gamma
                    relative=(candidate-target).square().sum(-1).sqrt()/target.square().sum(-1).sqrt().clamp_min(1e-30)
                    baseline_q=torch.round(x*(1<<fraction))
                    stats=dict(corpus=label,offset=offset,layer=layer,fraction_bits=fraction,queries=length,
                        denominator_zero=int((dq==0).sum()),invalid_norm=int((~valid).sum()),
                        norm_accumulator_exceeds_signed63=int((u.square().sum(-1)>=float(1<<62)).sum()),
                        baseline_norm_accumulator_exceeds_signed63=int((baseline_q.square().sum(-1)>=float(1<<62)).sum()),
                        numerator_exceeds_signed63=int((u.abs().amax(-1)>=float(1<<62)).sum()),
                        numerator_exceeds_exact_float64_integer=int((u.abs().amax(-1)>float(1<<53)).sum()),
                        max_relative_norm_output_error=float(relative.max()),mean_relative_norm_output_error=float(relative.mean()),
                        max_abs_quantized_numerator=float(u.abs().max()),min_centered_log2_denominator=float(centered[layer].min()),max_centered_log2_denominator=float(centered[layer].max()))
                    rows.append(stats)
            if index%8==7:print('finished',label,index+1,'windows',flush=True)
    path=ROOT/'results/projective_norm_denominators.npz'
    np.savez_compressed(path,calibration_logd=np.stack(calibration),public_gauge_log2=gauge,
        wiki_logd=np.stack([v['logd'] for v in log_captures if v['corpus']=='wiki']),ptb_logd=np.stack([v['logd'] for v in log_captures if v['corpus']=='ptb']))
    result=dict(records=rows,all_real_identity_checks_passed=True,heads=model.config.num_attention_heads,kv_heads=model.config.num_key_value_heads,
        hidden=model.config.hidden_size,query_length=length,calibration_offsets=ranges['calibration_offsets'],
        gauge='Integer per-layer/per-query-position median log2 product denominator from4 frozen WikiText2 calibration windows; same gauge for64 heldout windows.',
        denominator_definition='Z_h=1/max softmax_probability_h, corresponding to max-shifted exponent denominator; D=product_h Z_h. Collected from original FP32 full model, not from the finite SiLU proxy.',
        exact_identity='RMSNorm(u/D)=u/sqrt(mean(u²)+epsilon*D²),D>0. Multihead residual u=D*x+sum_h(product_{g!=h}Z_g)*W_h*N_h.',
        worst_case_range='With max-shifted exponentials,1<=Z_h<=causal_length. Raw exact product of integer exponent denominators at scale2^f needs up to H*(f+ceil(log2 L)) bits; choosing a public gauge shifts magnitude, not the input-domain dynamic range.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        corpus_hashes={label:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for label,name in [('wiki','wikitext2_test.txt'),('ptb','ptb_test.txt')]},
        limits='Representation-only necessary screen. It constructs u=x*D using already normalized attention residuals, so omits errors/costs of constructing u from private unnormalized per-head numerators. Float64 rounding/norm arithmetic is numerical, not exact large-integer certification. No private protocol, end-to-end model quality or novelty claim.')
    (ROOT/'results/projective_norm_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    for fraction in (16,24,32):
        subset=[v for v in rows if v['fraction_bits']==fraction];n=sum(v['queries'] for v in subset)
        print('fraction',fraction,'queries',n,'dq_zero',sum(v['denominator_zero'] for v in subset),'norm_overflow',sum(v['norm_accumulator_exceeds_signed63'] for v in subset),'baseline_overflow',sum(v['baseline_norm_accumulator_exceeds_signed63'] for v in subset),'max_error',max(v['max_relative_norm_output_error'] for v in subset),flush=True)
    for h in handles:h.remove()


if __name__=='__main__':main()
