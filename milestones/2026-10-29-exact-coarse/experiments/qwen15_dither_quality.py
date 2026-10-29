"""FP32 Qwen1.5B proxy; CPU int16 down weights, one transient GPU FP64 matrix.

Calibration is frozen before evaluation. No protocol timing or private-LLM claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from dithered_silu import build

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/Qwen2.5-1.5B'
REV='8faed761d45a263340a0528343f099c05c9a4323'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('aligned','dither','coarsebody'),default='dither')
    parser.add_argument('--corpus',choices=('wiki','ptb'),default='wiki');parser.add_argument('--windows',type=int,choices=(4,32),default=4)
    parser.add_argument('--seed',type=int,default=20261029);parser.add_argument('--shift',type=int,choices=(4,6,8),default=8)
    parser.add_argument('--output-fraction',type=int,choices=(0,8),default=0,help='Keep extra fractional bits in the signed64 down output.')
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((MODEL/'manifest.json').read_text());assert manifest['revision']==REV
    assert manifest['official_lfs_sha256']=='a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796'
    data=ROOT/'data/wikitext2_test.txt';data_sha=hashlib.sha256(data.read_bytes()).hexdigest()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);tokenizer.model_max_length=1<<30
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    mlps=[v.mlp for v in model.model.layers];assert len(mlps)==28
    tokens=tokenizer(data.read_text(),return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    range_path=ROOT/'results/qwen15_gate_ranges.json'
    if range_path.exists():
        calibration=json.loads(range_path.read_text());assert calibration['data_sha256']==data_sha and calibration['revision']==REV
    else:
        amax=[torch.zeros((),device='cuda') for _ in mlps];bmax=[torch.zeros((),device='cuda') for _ in mlps];hooks=[]
        def hook(values,i):
            def capture(module,inputs,out):values[i]=torch.maximum(values[i],out.detach().abs().max())
            return capture
        for i,m in enumerate(mlps):hooks.extend([m.gate_proj.register_forward_hook(hook(amax,i)),m.up_proj.register_forward_hook(hook(bmax,i))])
        with torch.inference_mode():
            for offset in (0,512,1024,1536):model(tokens[offset:offset+129][None].to('cuda'),use_cache=False)
        for h in hooks:h.remove()
        calibration=dict(revision=REV,data_sha256=data_sha,offsets=[0,512,1024,1536],length=129,
            gate_abs_max=[float(v) for v in amax],up_abs_max=[float(v) for v in bmax],dtype='float32')
        range_path.write_text(json.dumps(calibration,indent=2)+'\n')
    sa=[float(torch.tensor(v,dtype=torch.float32)/127) for v in calibration['gate_abs_max']]
    sb=[float(torch.tensor(v,dtype=torch.float32)/32767) for v in calibration['up_abs_max']]
    weights=[];l1=[];selected={}
    for i,m in enumerate(mlps):
        assert m.down_proj.bias is None
        q=torch.round(m.down_proj.weight.detach().cpu().double()*4096).to(torch.int64)
        assert int(q.min())>=-32768 and int(q.max())<=32767
        l1.append(int(q.abs().sum(1).max()));weights.append(q.to(torch.int16))
        if i in (0,14,27):selected[f'layer_{i}']=weights[-1].numpy()
    # Choose precision by a public all-domain certificate, never by test loss.
    chosen=None
    for fraction in range(22,15,-1):
        specs=[build(s,args.shift,fraction,33) for s in sa];cert=[]
        for i,s in enumerate(specs):
            g=s['absolute_bound'];full=l1[i]*g*32768;hi=l1[i]*((g+16383)//16384)*32768;lo=l1[i]*16383*32768
            cert.append(dict(layer=i,weight_integer_l1=l1[i],gate_absolute_bound=g,accumulator_bound=full,high_bound=hi,low_bound=lo))
        if all(c['accumulator_bound']<(1<<62) and c['high_bound']<(1<<53) and c['low_bound']<(1<<53) for c in cert):chosen=fraction;break
    assert chosen is not None
    final_shift=chosen+12-args.output_fraction
    for spec in specs:spec['output_fraction_bits']=args.output_fraction
    print('calibrated fraction',chosen,'max accumulator',max(c['accumulator_bound'] for c in cert),'peakGPU',torch.cuda.max_memory_allocated(),flush=True)
    piece_tensors=[tuple(torch.tensor(s[k],device='cuda',dtype=torch.int64) for k in ('signed_starts','slopes','intercepts')) for s in specs]
    if args.corpus=='ptb':
        evaluation=ROOT/'data/ptb_test.txt';tokens=tokenizer(evaluation.read_text(),return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        offsets=list(range(0,args.windows*1024,1024));length=257
    else:evaluation=data;offsets=list(range(8192,8192+args.windows*512,512));length=129
    assert all(o+length<=len(tokens) for o in offsets)
    originals=[m.forward for m in mlps];rows=[];captured={};rng=torch.Generator(device='cuda').manual_seed(args.seed)
    for test_index,offset in enumerate(offsets):
        ids=tokens[offset:offset+length][None].to('cuda')
        with torch.inference_mode():baseline=model(ids,use_cache=False).logits[0,:-1].float()
        lp=baseline.log_softmax(-1);prob=lp.exp();target=ids[0,1:];stats=[dict(elements=0,classifier_changed=0,gate_clipped=0,up_clipped=0) for _ in mlps]
        def make_forward(i):
            def forward(self,x):
                a=self.gate_proj(x);b=self.up_proj(x)
                af=torch.round(a/sa[i]*256).to(torch.int64);bf=torch.round(b/sb[i]*256).to(torch.int64)
                assert min(int(af.min()),int(bf.min()))>=-(1<<31) and max(int(af.max()),int(bf.max()))<(1<<31)
                bq=torch.div(bf+128,256,rounding_mode='floor').clamp(-32767,32767)
                classifier=af;rho=None;bucket=1<<args.shift
                if args.mode!='aligned':
                    rho=torch.randint(0,bucket,af.shape,device='cuda',generator=rng)
                    classifier=(torch.div(af,bucket,rounding_mode='floor')+(rho<(af%bucket)).to(torch.int64))*bucket
                starts,slopes,intercepts=piece_tensors[i];index=torch.bucketize(classifier.contiguous(),starts[1:],right=True)
                body=classifier if args.mode=='coarsebody' else af;g=slopes[index]*body+intercepts[index]
                assert int(g.abs().max())<=cert[i]['gate_absolute_bound']
                high=torch.div(g,16384,rounding_mode='floor');low=g-high*16384
                w=weights[i].to(device='cuda',dtype=torch.float64)
                ah=((high*bq).double()@w.T).to(torch.int64);al=((low*bq).double()@w.T).to(torch.int64)
                acc=ah*16384+al;out=torch.div(acc,1<<final_shift,rounding_mode='floor')
                stats[i]['elements']+=a.numel();stats[i]['gate_clipped']+=int((af.abs()>127*256).sum());stats[i]['up_clipped']+=int((b.abs()>sb[i]*32767).sum())
                stats[i]['classifier_changed']+=int((index!=torch.bucketize(af.contiguous(),starts[1:],right=True)).sum())
                if test_index==0 and i in (0,14,27):
                    for k,v in [('a_fixed32',af),('b_fixed32',bf),('gate_numerator',g),('contraction_output',out)]:captured[f'layer_{i}_{k}']=v[0,0].cpu().numpy()
                    if rho is not None:captured[f'layer_{i}_public_dither']=rho[0,0].cpu().numpy()
                return out.float()*(sb[i]/(1<<args.output_fraction))
            return forward
        for i,m in enumerate(mlps):m.forward=types.MethodType(make_forward(i),m)
        with torch.inference_mode():candidate=model(ids,use_cache=False).logits[0,:-1].float()
        for m,original in zip(mlps,originals):m.forward=original
        lq=candidate.log_softmax(-1);j=torch.arange(len(target),device='cuda')
        row=dict(offset=offset,baseline_nll=float(-lp[j,target].mean()),candidate_nll=float(-lq[j,target].mean()),
            mean_kl=float((prob*(lp-lq)).sum(-1).mean()),top1_agreement=float((baseline.argmax(-1)==candidate.argmax(-1)).float().mean()),layer_stats=stats)
        rows.append(row);print(offset,'KL',row['mean_kl'],'top1',row['top1_agreement'],'NLLdelta',row['candidate_nll']-row['baseline_nll'],flush=True)
    label=f'qwen15_dither_quality_{args.mode}_b{args.shift}_s{args.seed}_{args.corpus}{args.windows}'
    if args.output_fraction:label+=f'_outf{args.output_fraction}'
    capture=ROOT/'results'/f'{label}_capture.npz';np.savez_compressed(capture,**captured)
    wpath=ROOT/'results/qwen15_down_int12_selected.npz'
    if not wpath.exists():np.savez_compressed(wpath,**selected)
    out=dict(model_revision=REV,mode=args.mode,classifier_shift=args.shift,dither_seed=args.seed if args.mode!='aligned' else None,
        fraction_bits=chosen,output_fraction_bits=args.output_fraction,final_shift=final_shift,specifications=specs,integer_certificates=cert,records=rows,predicted_positions=args.windows*(length-1),
        evaluation_corpus=str(evaluation.relative_to(ROOT)),evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(range_path.read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        selected_down_sha256=hashlib.sha256(wpath.read_bytes()).hexdigest(),dtype='float32',hardware=torch.cuda.get_device_name(),gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        memory_strategy='Original modelFP32 onGPU. Down matrices quantizedint16 onCPU; only current layer expanded toGPUFP64. Same exact split integer dot arithmetic as0.5B proxy.',
        limits='All28 gate/down finite; upstream projections,attention,norm float32. Fixed calibration,reset-context windows; not fullcorpusPPL or private protocol timing/security/novelty.')
    (ROOT/'results'/f'{label}.json').write_text(json.dumps(out,indent=2)+'\n')
    print('meanKL',np.mean([r['mean_kl'] for r in rows]),'top1',np.mean([r['top1_agreement'] for r in rows]),flush=True)


if __name__=='__main__':main()
