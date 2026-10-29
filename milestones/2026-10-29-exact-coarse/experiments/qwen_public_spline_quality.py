"""All-layer GPU numerical proxy for public affine SiLU, exact finite down dot."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from public_silu_spline import build

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/Qwen2.5-0.5B';DATA=ROOT/'data/wikitext2_test.txt'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extended-test',action='store_true')
    parser.add_argument('--corpus-path',type=Path);parser.add_argument('--evaluation-label')
    parser.add_argument('--windows',type=int,default=32);parser.add_argument('--start-offset',type=int,default=0)
    parser.add_argument('--stride',type=int,default=512);parser.add_argument('--length',type=int)
    parser.add_argument('--classifier-shift',type=int,choices=(0,4,6,8),default=0)
    parser.add_argument('--exact-aligned',action='store_true');parser.add_argument('--dither-seed',type=int,default=20261029)
    parser.add_argument('--coarse-body',action='store_true',help='Ablation: evaluate the polynomial on the rounded classifier input too.')
    parser.add_argument('--output-fraction',type=int,choices=(0,8),default=0)
    parser.add_argument('--fraction-bits',type=int,default=22);args=parser.parse_args()
    data_path=DATA if args.corpus_path is None else args.corpus_path.resolve()
    assert not args.coarse_body or (args.classifier_shift and not args.exact_aligned)
    if args.corpus_path is not None:
        assert args.evaluation_label and args.evaluation_label.replace('_','').isalnum()
        assert not args.extended_test and args.windows>0 and args.start_offset>=0 and args.stride>0
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ranges=json.loads((ROOT/'results/qwen_gate_ranges.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==ranges['data_sha256']
    amax=torch.tensor(ranges['gate_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values
    bmax=torch.tensor(ranges['up_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values
    sa=amax/127;sb=bmax/32767
    if args.classifier_shift:
        from dithered_silu import build as dither_build
        specifications=[dither_build(float(s),args.classifier_shift,args.fraction_bits,33) for s in sa.cpu()]
    else:
        assert not args.exact_aligned
        specifications=[build(float(s),args.fraction_bits) for s in sa.cpu()]
    dither_generator=torch.Generator(device='cuda').manual_seed(args.dither_seed)
    for specification in specifications:specification['output_fraction_bits']=args.output_fraction
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    tokenizer.model_max_length=1<<30
    tokens=tokenizer(data_path.read_text(),return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    mlps=[layer.mlp for layer in model.model.layers];originals=[m.forward for m in mlps]
    weights=[];piece_tensors=[];certificates=[]
    for i,m in enumerate(mlps):
        assert m.down_proj.bias is None
        w=torch.round(m.down_proj.weight.double()*4096).to(torch.int64)
        assert int(w.min())>=-32768 and int(w.max())<=32767
        l1=int(w.abs().sum(dim=1).max());spec=specifications[i];gmax=spec['absolute_bound']
        full=l1*gmax*32768;high=l1*((gmax+16383)//16384)*32768;low=l1*16383*32768
        assert full<(1<<62) and high<(1<<53) and low<(1<<53)
        certificates.append(dict(layer=i,gate_absolute_bound=gmax,maximum_weight_integer_l1=l1,
            full_absolute_accumulator_bound=full,high_dot_absolute_bound=high,low_dot_absolute_bound=low,
            signed63_safe=True,split_float64_exact=True,private_b_value_bits=16,
            final_fraction_removed=args.fraction_bits+12-args.output_fraction))
        weights.append(w.double())
        piece_tensors.append(tuple(torch.tensor(spec[k],device='cuda',dtype=torch.int64) for k in ('signed_starts','slopes','intercepts')))
    rows=[];captured={};offsets=list(range(8192,8192+32*512,512)) if args.extended_test else ranges['test_offsets']
    length=args.length or ranges['length'];assert length>=2
    if args.corpus_path is not None:offsets=list(range(args.start_offset,args.start_offset+args.windows*args.stride,args.stride))
    assert all(offset+length<=len(tokens) for offset in offsets)
    for test_index,offset in enumerate(offsets):
        ids=tokens[offset:offset+length][None].to('cuda')
        with torch.inference_mode():baseline=model(ids,use_cache=False).logits[0,:-1].float()
        logp=baseline.log_softmax(-1);prob=logp.exp();target=ids[0,1:]
        stats=[dict(elements=0,gate_clipped=0,up_clipped=0,classifier_changed=0,silu_error_square_sum=0.0,silu_max_abs_error=0.0) for _ in mlps]
        def make_forward(i):
            def forward(self,x):
                a=self.gate_proj(x);b=self.up_proj(x)
                af=torch.round(a/sa[i]*256).to(torch.int64);bf=torch.round(b/sb[i]*256).to(torch.int64)
                assert min(int(af.min()),int(bf.min()))>=-(1<<31) and max(int(af.max()),int(bf.max()))<(1<<31)
                bq=torch.div(bf+128,256,rounding_mode='floor').clamp(-32767,32767)
                starts,slopes,intercepts=piece_tensors[i]
                classifier=af;rho=None
                if args.classifier_shift and not args.exact_aligned:
                    bucket=1<<args.classifier_shift;rho=torch.randint(0,bucket,af.shape,device='cuda',generator=dither_generator)
                    classifier=(torch.div(af,bucket,rounding_mode='floor')+(rho<(af%bucket)).to(torch.int64))*bucket
                index=torch.bucketize(classifier.contiguous(),starts[1:].contiguous(),right=True)
                stats[i]['classifier_changed']+=int((index!=torch.bucketize(af.contiguous(),starts[1:].contiguous(),right=True)).sum())
                body=classifier if args.coarse_body else af
                g=slopes[index]*body+intercepts[index]
                assert int(g.abs().max())<=certificates[i]['gate_absolute_bound']
                high=torch.div(g,16384,rounding_mode='floor');low=g-high*16384
                ah=((high*bq).double()@weights[i].T).to(torch.int64)
                al=((low*bq).double()@weights[i].T).to(torch.int64)
                acc=ah*16384+al
                result=torch.div(acc,1<<(args.fraction_bits+12-args.output_fraction),rounding_mode='floor')
                err=g.double()/(1<<args.fraction_bits)-torch.nn.functional.silu(a).double()
                stats[i]['elements']+=a.numel();stats[i]['gate_clipped']+=int((af.abs()>127*256).sum())
                stats[i]['up_clipped']+=int((b.abs()>sb[i]*32767).sum())
                stats[i]['silu_error_square_sum']+=float(err.square().sum())
                stats[i]['silu_max_abs_error']=max(stats[i]['silu_max_abs_error'],float(err.abs().max()))
                if test_index==0 and i in (0,12,23):
                    prefix=f'layer_{i}'
                    for key,value in (('a_fixed32',af),('b_fixed32',bf),('b_integer',bq),('gate_numerator',g),('contraction_output',result)):
                        captured[prefix+'_'+key]=value[0,0].cpu().numpy()
                    if rho is not None:captured[prefix+'_public_dither']=rho[0,0].cpu().numpy()
                return result.float()*(sb[i]/(1<<args.output_fraction))
            return forward
        for i,m in enumerate(mlps):m.forward=types.MethodType(make_forward(i),m)
        with torch.inference_mode():candidate=model(ids,use_cache=False).logits[0,:-1].float()
        for m,f in zip(mlps,originals):m.forward=f
        logq=candidate.log_softmax(-1);index=torch.arange(len(target),device='cuda')
        row=dict(offset=offset,baseline_nll=float(-logp[index,target].mean()),candidate_nll=float(-logq[index,target].mean()),
            mean_kl=float((prob*(logp-logq)).sum(-1).mean()),top1_agreement=float((baseline.argmax(-1)==candidate.argmax(-1)).float().mean()),layer_stats=stats)
        rows.append(row);print(offset,'KL',row['mean_kl'],'top1',row['top1_agreement'],'NLL delta',row['candidate_nll']-row['baseline_nll'],flush=True)
    suffix=f'f{args.fraction_bits}'+('_extended32' if args.extended_test else '')
    if args.classifier_shift:suffix+=f'_alignedb{args.classifier_shift}' if args.exact_aligned else f'_ditherb{args.classifier_shift}_seed{args.dither_seed}'
    if args.coarse_body:suffix+='_coarsebody'
    if args.output_fraction:suffix+=f'_outf{args.output_fraction}'
    if args.corpus_path is not None:suffix+='_'+args.evaluation_label
    capture=ROOT/'results'/f'qwen_public_spline_{suffix}_capture.npz';np.savez_compressed(capture,**captured)
    output=dict(model_revision='060db6499f32faf8b98477b0a26969ef7d8b9987',specifications=specifications,
        integer_certificates=certificates,records=rows,predicted_positions=len(offsets)*(length-1),
        evaluation_corpus=str(data_path.relative_to(ROOT)),evaluation_corpus_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        evaluation_window_length=length,evaluation_token_count=len(tokens),calibration_corpus_sha256=ranges['data_sha256'],
        classifier_shift=args.classifier_shift,exact_aligned=args.exact_aligned,dither_seed=args.dither_seed if args.classifier_shift and not args.exact_aligned else None,
        coarse_body_ablation=args.coarse_body,
        output_fraction_bits=args.output_fraction,
        dither_functionality='Public independent threshold rho; coarse index z=floor(A/B)+I(rho<AmodB), polynomial body uses original A.33-bit guard view, signed32 input; against exact32 classifier net bit saving b-1.' if args.classifier_shift and not args.exact_aligned else None,
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),range_sha256=hashlib.sha256((ROOT/'results/qwen_gate_ranges.json').read_bytes()).hexdigest(),
        hardware=torch.cuda.get_device_name(),gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        limits='Numerical proxy only. All24 gates and down projections are finite integer functions, other operations floating point. Fixed4 WikiText2 calibration windows; no new fitting to test contexts. Sampled windows with reset context, not full corpus PPL, private GPU inference or confirmed new FSS primitive.')
    (ROOT/'results'/f'qwen_public_spline_quality_{suffix}.json').write_text(json.dumps(output,indent=2)+'\n')
    print('mean KL',np.mean([r['mean_kl'] for r in rows]),'top1',np.mean([r['top1_agreement'] for r in rows]),flush=True)


if __name__=='__main__':main()
