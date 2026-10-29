"""Actual Qwen GPU gate proxy matching exact FSS floor postprocessing.

Reuses frozen calibration ranges and enumerated public integer LUTs. All model
matrix operations remain floating point; this is not end-to-end secure inference.
"""
import hashlib
import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/Qwen2.5-0.5B'
DATA=ROOT/'data/wikitext2_test.txt'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixed8-input-quantizer',action='store_true')
    parser.add_argument('--contraction-rounding',choices=('none','stage','late'),default='none')
    parser.add_argument('--extended-test',action='store_true')
    parser.add_argument('--gate-maximum',type=int,choices=(31,63,127),default=127)
    parser.add_argument('--keep-up-fraction',action='store_true');args=parser.parse_args()
    assert args.contraction_rounding=='none' or args.fixed8_input_quantizer
    assert not args.keep_up_fraction or (args.fixed8_input_quantizer and args.contraction_rounding=='late')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ranges=json.loads((ROOT/'results/qwen_gate_ranges.json').read_text())
    certificate=json.loads((ROOT/'results/qwen_gate_integer_certificate.json').read_text())
    assert certificate['all_layers_passed'] and hashlib.sha256(DATA.read_bytes()).hexdigest()==ranges['data_sha256']
    public_tables=np.load(ROOT/'results/qwen_layer_lut_tables.npz')['tables']
    amax=torch.tensor(ranges['gate_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values
    bmax=torch.tensor(ranges['up_max_per_channel'],dtype=torch.float32,device='cuda').max(dim=1).values
    sa=amax/args.gate_maximum;sb=bmax/32767
    if args.gate_maximum!=127:
        codes=torch.arange(256,device='cuda',dtype=torch.int64);signed=(codes^128)-128
        generated=torch.round(torch.nn.functional.silu(sa[:,None]*signed.float()[None,:])*256).to(torch.int64)
        generated[:,signed.abs()>args.gate_maximum]=0
        public_tables=generated.cpu().numpy()
    table_abs_bounds=[max(abs(int(v)) for v in table) for table in public_tables]
    assert max(table_abs_bounds)<=32767
    b_integer_bound=32768*(256 if args.keep_up_fraction else 1)
    tables=torch.tensor(public_tables,device='cuda',dtype=torch.int64)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    tokenizer.model_max_length=1<<30
    tokens=tokenizer(DATA.read_text(),return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    mlps=[layer.mlp for layer in model.model.layers];originals=[m.forward for m in mlps]
    contraction_certificates=[];contraction_weights=[]
    if args.contraction_rounding!='none':
        for i,m in enumerate(mlps):
            assert m.down_proj.bias is None
            w=torch.round(m.down_proj.weight.double()*4096).to(torch.int64)
            assert int(w.abs().max())<=32767
            l1=int(w.abs().sum(dim=1).max());bound=l1*table_abs_bounds[i]*b_integer_bound
            # Every integer product and every possible partial sum is exact in
            # float64 under this absolute-sum bound; no approximate dot product.
            assert bound<(1<<63)
            high_bound=l1*table_abs_bounds[i]*32768;low_bound=l1*table_abs_bounds[i]*255
            assert high_bound<(1<<53) and low_bound<(1<<53)
            assert args.keep_up_fraction or bound<(1<<53)
            contraction_certificates.append(dict(layer=i,weight_fraction_bits=12,max_abs_weight_integer=int(w.abs().max()),
                maximum_row_integer_l1=l1,absolute_accumulator_bound=bound,float64_integer_exact=bound<(1<<53),
                split_high_low_float64_integer_exact=True,high_term_bound=high_bound,low_term_bound=low_bound,signed64_safe=True))
            contraction_weights.append(w.double())
    rows=[];captured={}
    test_offsets=list(range(8192,8192+32*512,512)) if args.extended_test else ranges['test_offsets']
    assert max(test_offsets)+ranges['length']<=len(tokens)
    for test_index,offset in enumerate(test_offsets):
        ids=tokens[offset:offset+ranges['length']][None].to('cuda')
        with torch.inference_mode():baseline=model(ids,use_cache=False).logits[0,:-1].float()
        logp=baseline.log_softmax(-1);prob=logp.exp();target=ids[0,1:]
        stats=[dict(elements=0,gate_clipped=0,up_clipped=0,max_abs_raw=0,gate_codes_differ_from_direct_round=0,up_codes_differ_from_direct_round=0) for _ in mlps]
        def make_forward(i):
            def forward(self,x):
                a=self.gate_proj(x);b=self.up_proj(x)
                ai=torch.round(a/sa[i]).clamp(-args.gate_maximum,args.gate_maximum).to(torch.int64)
                bi=torch.round(b/sb[i]).clamp(-32767,32767).to(torch.int64)
                if args.fixed8_input_quantizer:
                    af=torch.round(a/sa[i]*256).to(torch.int64);bf=torch.round(b/sb[i]*256).to(torch.int64)
                    assert min(int(af.min()),int(bf.min()))>=-(1<<31)
                    assert max(int(af.max()),int(bf.max()))<(1<<31)
                    aq=torch.div(af+128,256,rounding_mode='floor').clamp(-args.gate_maximum,args.gate_maximum)
                    bq=torch.div(bf+128,256,rounding_mode='floor').clamp(-32767,32767)
                    stats[i]['gate_codes_differ_from_direct_round']+=int((aq!=ai).sum())
                    stats[i]['up_codes_differ_from_direct_round']+=int((bq!=bi).sum())
                    ai=aq;bi=bq
                    if args.keep_up_fraction:bi=bf.clamp(-32767*256,32767*256)
                g=tables[i][ai&255];raw=g*bi
                q=torch.div(raw,256,rounding_mode='floor')
                assert int(raw.abs().max())<=table_abs_bounds[i]*b_integer_bound
                stats[i]['elements']+=a.numel();stats[i]['gate_clipped']+=int((a.abs()>sa[i]*args.gate_maximum).sum())
                stats[i]['up_clipped']+=int((b.abs()>sb[i]*32767).sum())
                stats[i]['max_abs_raw']=max(stats[i]['max_abs_raw'],int(raw.abs().max()))
                if test_index==0 and i in (0,12,23):
                    prefix=f'layer_layer_{i}'
                    captured[prefix+'_a_index']=((ai[0,0]&255).cpu().numpy()).astype(np.uint8)
                    captured[prefix+'_b_integer']=bi[0,0].cpu().numpy()
                    captured[prefix+'_lookup']=public_tables[i]
                    captured[prefix+'_up_scale']=sb[i].cpu().numpy()
                    captured[prefix+'_gate_scale']=sa[i].cpu().numpy()
                    captured[prefix+'_a_real']=a[0,0].cpu().numpy();captured[prefix+'_b_real']=b[0,0].cpu().numpy()
                    captured[prefix+'_a_fixed32']=torch.round(a[0,0]/sa[i]*256).to(torch.int64).cpu().numpy()
                    captured[prefix+'_b_fixed32']=torch.round(b[0,0]/sb[i]*256).to(torch.int64).cpu().numpy()
                    captured[prefix+'_floor_output']=q[0,0].cpu().numpy()
                if args.contraction_rounding=='none':return self.down_proj(q.float()*sb[i])
                operand=q if args.contraction_rounding=='stage' else raw
                if args.keep_up_fraction:
                    high=torch.div(bi,256,rounding_mode='floor');low=bi-high*256
                    ah=((g*high).double()@contraction_weights[i].T).to(torch.int64)
                    al=((g*low).double()@contraction_weights[i].T).to(torch.int64)
                    acc=ah*256+al
                else:acc=(operand.double()@contraction_weights[i].T).to(torch.int64)
                divisor=4096 if args.contraction_rounding=='stage' else 4096*256
                if args.keep_up_fraction:divisor*=256
                contracted=torch.div(acc,divisor,rounding_mode='floor')
                return contracted.float()*sb[i]
            return forward
        for i,m in enumerate(mlps):m.forward=types.MethodType(make_forward(i),m)
        with torch.inference_mode():candidate=model(ids,use_cache=False).logits[0,:-1].float()
        for m,f in zip(mlps,originals):m.forward=f
        logq=candidate.log_softmax(-1);index=torch.arange(len(target),device='cuda')
        row={'offset':offset,'baseline_nll':float(-logp[index,target].mean()),'candidate_nll':float(-logq[index,target].mean()),
             'mean_kl':float((prob*(logp-logq)).sum(-1).mean()),
             'top1_agreement':float((baseline.argmax(-1)==candidate.argmax(-1)).float().mean()),'layer_stats':stats}
        rows.append(row);print(offset,'NLL delta',row['candidate_nll']-row['baseline_nll'],'KL',row['mean_kl'],flush=True)
    suffix='fixed8_input' if args.fixed8_input_quantizer else 'faithful'
    if args.contraction_rounding!='none':suffix+='_'+args.contraction_rounding
    if args.gate_maximum!=127:suffix+=f'_g{args.gate_maximum}'
    if args.keep_up_fraction:suffix+='_upfraction'
    if args.extended_test:suffix+='_extended32'
    path=ROOT/'results'/f'qwen_gate_{suffix}_capture.npz';np.savez_compressed(path,**captured)
    out={'model':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
         'postprocessing':f'exact late contraction floor(sum(Wq*g*b)/{4096*256*(256 if args.keep_up_fraction else 1)})' if args.contraction_rounding=='late' else 'stagewise floor(product/256), then optional fixed12 down-contraction floor',
         'gate_symmetric_maximum':args.gate_maximum,'up_branch_retains_fixed8_fraction':args.keep_up_fraction,
         'public_table_sha256':hashlib.sha256(public_tables.astype('<i8').tobytes()).hexdigest(),
         'input_quantization':'normalized fixed8 followed by half-up rounding and symmetric clipping' if args.fixed8_input_quantizer else 'direct float ties-to-even quantization',
         'contraction_rounding':args.contraction_rounding,'contraction_integer_certificates':contraction_certificates,
         'test_offsets':test_offsets,'predicted_positions':len(test_offsets)*(ranges['length']-1),
         'records':rows,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'range_file_sha256':certificate['range_file_sha256'],'gpu_peak_bytes':torch.cuda.max_memory_allocated(),
         'hardware':torch.cuda.get_device_name(),'torch':torch.__version__,
         'limits':f'All24 gated MLPs changed, other model operations floating point. Frozen4 calibration windows,{len(test_offsets)} test windows,{len(test_offsets)*(ranges["length"]-1)} predictions; not corpus PPL or secure end-to-end model. Encoding secret matrix accumulation into normalized fixed8 input shares is outside the implemented operator protocol. No stochastic local truncation in this proxy.'}
    (ROOT/'results'/f'qwen_gated_quality_{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
    print('mean KL',np.mean([r['mean_kl'] for r in rows]),'top1',np.mean([r['top1_agreement'] for r in rows]),flush=True)


if __name__=='__main__':main()
