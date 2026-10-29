"""Real4864 gates; GPU individual-party checks and resident gated arithmetic.

Both parties execute sequentially on one GPU. Masked openings are arithmetic
in a trusted process, not communication. No private down matrix is included.
"""
import hashlib
import json
import time
import numpy as np
import torch
from fused_gated_lut import ROOT,random64,split64
from public_silu_spline import build,evaluate as plain
from public_spline_coefficient_basis import prepare as basis_prepare,evaluate as basis_evaluate
from public_spline_selector import prepare as selector_prepare,evaluate as selector_evaluate
from public_spline_cuda import prepare_runtime,evaluate,raw,MASK
from dcf_cuda import tensor,LIB


def measure(fn,repeats=12):
    for _ in range(3):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
    return dict(median_ms=float(np.median(times)),quartiles_ms=np.quantile(times,[.25,.75]).tolist(),records_ms=times)


def main():
    torch.set_num_threads(1);rows=[];quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text())
    rng=np.random.default_rng(20261029)
    with np.load(ROOT/'results/qwen_public_spline_f22_capture.npz') as data:
        for layer in (0,12,23):
            spec=build(quality['specifications'][layer]['sa'],22,32);a=data[f'layer_{layer}_a_fixed32'];b=data[f'layer_{layer}_b_integer'];n=len(a)
            alpha=random64((n,))&np.uint64(MASK);opened=(a.view(np.uint64)+alpha)&np.uint64(MASK);items={}
            for variant in ('selector','basis','weighted','wide'):
                start=time.perf_counter_ns()
                if variant in ('selector','weighted'):keys,_=selector_prepare(n,spec,63,alpha,variant=='weighted')
                else:keys,_=basis_prepare(n,spec,63,alpha,variant)
                dealer_ms=(time.perf_counter_ns()-start)/1e6
                start=time.perf_counter_ns();runtime=[prepare_runtime(keys[p],opened,spec,variant) for p in (0,1)];torch.cuda.synchronize()
                upload_compile_ms=(time.perf_counter_ns()-start)/1e6
                for p in (0,1):
                    gpu=evaluate(runtime[p],p).cpu().numpy().view(np.uint64)
                    cpu=selector_evaluate(keys[p],opened,spec,p,63,variant=='weighted') if variant in ('selector','weighted') else basis_evaluate(keys[p],opened,spec,p,63,cached=3,variant=variant)
                    assert np.array_equal(gpu,cpu),(layer,variant,p)
                if variant=='weighted':rb=(keys[0]['b_mask']+keys[1]['b_mask'])&np.uint64(MASK);mult=None
                else:
                    rb=random64((n,))&np.uint64(MASK);rg=random64((n,))&np.uint64(MASK)
                    rbs=split64(rb);rgs=split64(rg);prod=split64(rb*rg)
                    mult=[dict(rb=tensor(rbs[p]),rg=tensor(rgs[p]),product=tensor(prod[p])) for p in (0,1)]
                bh=tensor((b.view(np.uint64)+rb)&np.uint64(MASK))
                def gated(runtime=runtime,variant=variant,mult=mult,bh=bh):
                    values=[evaluate(runtime[p],p) for p in (0,1)]
                    if variant=='weighted':return [(bh*v[:,0]+v[:,1])&MASK for v in values]
                    if variant=='selector':
                        sh=(values[0][:,0]+runtime[0]['key']['horner_rs']+values[1][:,0]+runtime[1]['key']['horner_rs'])&MASK
                        gates=[]
                        for p in (0,1):
                            k=runtime[p]['key'];h=runtime[p]['h']
                            g=k['horner_product']-h*k['horner_rs']-sh*k['input_mask']+values[p][:,1]+k['refresh']
                            if p==0:g=g+h*sh
                            gates.append(g&MASK)
                    else:gates=values
                    gh=(gates[0]+mult[0]['rg']+gates[1]+mult[1]['rg'])&MASK
                    answer=[mult[p]['product']-gh*mult[p]['rb']-bh*mult[p]['rg'] for p in (0,1)]
                    answer[0]=answer[0]+gh*bh
                    return [v&MASK for v in answer]
                answer=gated();actual=((answer[0]+answer[1])&MASK).cpu().numpy().view(np.uint64)
                expected=(plain(spec,a)*b).view(np.uint64)&np.uint64(MASK);assert np.array_equal(actual,expected)
                items[variant]=dict(runtime=runtime,gated=gated,dealer_ms=dealer_ms,upload_compile_ms=upload_compile_ms)
                print('checked layer',layer,variant,'individual parties and gated output',flush=True)
            variants=('selector','basis','weighted','wide')
            for repeat in range(4):
                order=variants[repeat:]+variants[:repeat]
                for variant in order:
                    item=items[variant];runtime=item['runtime']
                    t=measure(item['gated'])
                    row=dict(layer=layer,variant=variant,repeat=repeat,order=list(order),n=n,m=len(spec['unsigned_pieces']),gated_both_parties_sequential=t,
                        dealer_ms=item['dealer_ms'],upload_and_public_compile_ms=item['upload_compile_ms'],all_checks_passed=True)
                    rows.append(row)
            print('finished layer',layer,flush=True)
            del items
    out=dict(records=rows,all_passed=True,hardware=torch.cuda.get_device_name(),native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        timing='Resident GPU, one thread per public interval query, no prefix reuse, CUDA-event timing. Both parties sequential; trusted process adds masked shares to model openings.12 repeats per each of four cyclic orders. Input queries and public coefficient weights are compiled before timing.',
        limits='Research software AES backend; not upstream GPU baseline. No socket/PCIe online transport, private down projection, B quantizer, final truncation or full secure LLM timing. Repeated keys/openings are for kernel timing only, not authorized protocol key reuse across requests.')
    (ROOT/'results/public_spline_cuda_benchmark.json').write_text(json.dumps(out,indent=2)+'\n')
    for variant in variants:
        print(variant,'median ms',float(np.median([v['gated_both_parties_sequential']['median_ms'] for v in rows if v['variant']==variant])),flush=True)


if __name__=='__main__':main()
