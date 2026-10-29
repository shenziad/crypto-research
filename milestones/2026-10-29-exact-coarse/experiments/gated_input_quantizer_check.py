"""Exhaustive small-ring identities and actual native two-party gate pipeline."""
import hashlib
import itertools
import json
import numpy as np
from gated_lut_strong_baselines import ROOT,run
from dpf_aes_native import LIB


def exhaustive():
    w,s=8,3;R=1<<w;D=1<<s;count=0
    for r in range(R):
        for x in range(R):
            h=(x+r)%R;theta=(r+R//2)%R
            got=h//D-r//D+(1 if r%D<D//2 else 0)-(h%D<(r%D+D//2)%D)+(R//D)*((h<theta)-(r<R//2))
            signed=x if x<R//2 else x-R
            assert got==(signed+D//2)//D
            count+=1
    clamp_checks=0
    for maximum in (1,3,31):
        for r in range(R):
            z=(R-1-r)%R;zs=z if z<R//2 else z-R
            slope,intercept=(1,-r) if -maximum<=zs<maximum else (0,maximum if zs>=0 else -maximum)
            terms=[((r+maximum)%R,1,-maximum-r),((r+R//2)%R,0,2*maximum),((r-maximum)%R,-1,r-maximum)]
            for h in range(R):
                coeff=slope;constant=intercept
                for theta,dm,dc in terms:
                    if h<theta:coeff+=dm;constant+=dc
                got=(coeff*h+constant)%R
                x=(h-r)%R;signed=x if x<R//2 else x-R
                assert got==min(max(signed,-maximum),maximum)%R
                clamp_checks+=1
    return dict(rounding_pairs=count,clipping_pairs=clamp_checks,all_passed=True)


def main():
    small=exhaustive();print(small,flush=True)
    rng=np.random.default_rng(2026102908)
    # Full signed32 extremes plus every remainder near both saturation bounds.
    edge=np.concatenate([np.arange(-384,385),
        np.array([-(1<<31),-(1<<31)+1,(1<<31)-2,(1<<31)-1]),
        np.concatenate([m*256+np.arange(-384,385) for m in (-32767,-127,127,32767)])])
    a=np.concatenate([edge,rng.integers(-(1<<31),1<<31,4096,dtype=np.int64)]).astype(np.int64).view(np.uint64)
    b=a[::-1].copy();checks=[]
    for mode in ('beaver32','mixed16','fused32'):
        checks.append(run(a,b,mode,input_fixed32=True));print('functional',mode,len(a),flush=True)
    capture=ROOT/'results/qwen_gate_faithful_capture.npz';rows=[]
    orders=list(itertools.permutations(('beaver32','mixed16','fused32')))
    with np.load(capture) as data:
        for layer in (0,12,23):
            prefix=f'layer_layer_{layer}'
            a=data[prefix+'_a_fixed32'].view(np.uint64);b=data[prefix+'_b_fixed32'].view(np.uint64)
            table=data[prefix+'_lookup'].view(np.uint64)
            for rtt in (0,50):
                for repetition,order in enumerate(orders):
                    for mode in order:
                        row=run(a,b,mode,table,rtt_ms=rtt,input_fixed32=True)
                        row.update(layer=layer,repetition=repetition,order=list(order));rows.append(row)
                print('real layer',layer,'emulated RTT',rtt,flush=True)
    out=dict(small_ring_identity_checks=small,full32_boundary_random_checks=checks,
        results=rows,capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        input_contract='normalized signed32 fixed8 inputs; quantizer is clip(floor((signed(x)+128)/256), symmetric bounds127/32767)',
        limits='Existing DCF and piecewise-affine offset FSS; no novelty claim. Local trusted parent/fork and UNIX sockets. All projections and conversion from secret matrix accumulation scale to specified normalized fixed8 shares remain outside. Emulated RTT is not WAN. Dealer network and harness reconstruction excluded.')
    (ROOT/'results/gated_input_quantizer_pipeline.json').write_text(json.dumps(out,indent=2)+'\n')
    for rtt in (0,50):
        for mode in ('beaver32','mixed16','fused32'):
            r=[v for v in rows if v['mode']==mode and v['emulated_rtt_ms_per_logical_exchange']==rtt]
            print(rtt,mode,'ms',np.median([v['parties'][0]['total_ms'] for v in r]),
                'bytes',r[0]['online_sent_bytes_both_parties'],'keys',r[0]['total_key_array_bytes_both_parties'],flush=True)


if __name__=='__main__':main()
