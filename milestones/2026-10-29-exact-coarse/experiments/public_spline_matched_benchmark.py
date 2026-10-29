"""Matched long-payload versus public-basis affine FSS, same ring63 function."""
import hashlib
import json
import time
import numpy as np
from fused_gated_lut import ROOT,random64
from public_silu_spline import build,evaluate as plain
from public_spline_coefficient_basis import prepare,compile_queries,compile_constants,evaluate
from dcf_wide_native import LIB as WIDE_LIB
from public_spline_fss import LIB as BASIS_LIB


def main():
    quality=json.loads((ROOT/'results/qwen_public_spline_quality_f22.json').read_text())
    capture=ROOT/'results/qwen_public_spline_f22_capture.npz';rows=[];mask=np.uint64((1<<63)-1)
    modes=(('basis',0),('wide',0),('basis',3),('wide',3))
    orders=[modes[i:]+modes[:i] for i in range(len(modes))]
    with np.load(capture) as data:
        for layer in (0,12,23):
            a=data[f'layer_{layer}_a_fixed32'];spec=build(quality['specifications'][layer]['sa'],22,63)
            n=len(a);alpha=random64((n,))&mask;opened=(a.view(np.uint64)+alpha)&mask
            expected=plain(spec,a).view(np.uint64)&mask;assert np.array_equal(plain(spec,a),data[f'layer_{layer}_gate_numerator'])
            keys={};setup={};compiled={};public_compile={}
            for variant in ('wide','basis') if layer==12 else ('basis','wide'):
                start=time.perf_counter_ns();keys[variant],_=prepare(n,spec,63,alpha,variant)
                setup[variant]=(time.perf_counter_ns()-start)/1e6
                for mode in (0,3):
                    start=time.perf_counter_ns();compiled[(variant,mode)]=(compile_queries if variant=='basis' and mode==0 else compile_constants)(spec,opened,63)
                    public_compile[(variant,mode)]=(time.perf_counter_ns()-start)/1e6
            for repeat,order in enumerate(orders):
                for variant,cached in order:
                    start=time.perf_counter_ns();values=[evaluate(keys[variant][p],opened,spec,p,63,compiled[(variant,cached)],cached,variant) for p in (0,1)]
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    assert np.array_equal((values[0]+values[1])&mask,expected)
                    rows.append(dict(layer=layer,n=n,variant=variant,native_cache_mode=cached,repeat=repeat,
                        order=[f'{v}:{c}' for v,c in order],both_party_sequential_ms=elapsed,
                        dealer_ms=setup[variant],public_compile_ms=public_compile[(variant,cached)],
                        key_array_bytes_both_parties=sum(v.nbytes for k in keys[variant] for v in k.values()),
                        all_passed=True,public_pieces=len(spec['slopes'])))
            print('finished real layer',layer,flush=True)
    out=dict(results=rows,description='Same public integer-affine SiLU, input/output ring63, same captured normalized A values. Both variants compress public constants; wide baseline additionally uses partial coordinates and public-prefix aggregate evaluation.',
        wide_native_sha256=hashlib.sha256(WIDE_LIB.read_bytes()).hexdigest(),basis_native_sha256=hashlib.sha256(BASIS_LIB.read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        limits='Our matched AES-NI backend, not execution of original LLAMA fixed-key MMO library. Local sequential kernels, no network or B quantization/private matrix layer. Each mode occurs once in each timing position; not every permutation. Same key/opening reused only for identical implementation timing. No independent novelty or full secure model speedup claim.')
    (ROOT/'results/public_spline_matched_benchmark.json').write_text(json.dumps(out,indent=2)+'\n')
    for layer in (0,12,23):
        for variant,cached in modes:
            r=[v for v in rows if v['layer']==layer and v['variant']==variant and v['native_cache_mode']==cached]
            print(layer,variant,cached,'median ms',np.median([v['both_party_sequential_ms'] for v in r]),'keys',r[0]['key_array_bytes_both_parties'],flush=True)


if __name__=='__main__':main()
