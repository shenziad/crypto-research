"""Independent AES/key reference plus wide partial/prefix FSS checks."""
import hashlib
import json
import os
import numpy as np
from dcf_aes_native_check import aes
from dcf_aes_native import generate_batch as small_gen
from dcf_wide_native import generate,select,LIB
from fused_gated_lut import ROOT,random64
from public_spline_coefficient_basis import prepare,evaluate


def reference(alpha,beta,bits,roots):
    dim=len(beta);blocks=(dim+1)//2;mask=(1<<64)-1
    seeds=[roots[:16],roots[16:]];control=[0,1];along=[0]*dim;corrections=[]
    def decode(raw):return [int.from_bytes(raw[j*8:(j+1)*8],'little') for j in range(dim)]
    for level in range(bits):
        raw=[aes(seed,range(3+2*blocks)) for seed in seeds]
        children=[[v[:16],v[16:32]] for v in raw];t=[[(v[32]&1),((v[32]>>1)&1)] for v in raw]
        values=[[decode(b''.join(v[(3+2*b)*16:(4+2*b)*16] for b in range(blocks))),
                 decode(b''.join(v[(4+2*b)*16:(5+2*b)*16] for b in range(blocks)))] for v in raw]
        keep=(alpha>>(bits-1-level))&1;lose=keep^1;sign=-1 if control[1] else 1
        vcw=[(sign*(-along[c]-values[0][lose][c]+values[1][lose][c]+(int(beta[c]) if keep else 0)))&mask for c in range(dim)]
        along=[(along[c]-values[1][keep][c]+values[0][keep][c]+sign*vcw[c])&mask for c in range(dim)]
        scw=bytes(a^b for a,b in zip(children[0][lose],children[1][lose]));tcw=[t[0][0]^t[1][0]^keep^1,t[0][1]^t[1][1]^keep]
        corrections.append(scw+bytes([tcw[0]|(tcw[1]<<1)])+b''.join(v.to_bytes(8,'little') for v in vcw))
        for p in (0,1):
            seeds[p]=bytes(a^b for a,b in zip(children[p][keep],scw)) if control[p] else children[p][keep]
            control[p]=t[p][keep]^(tcw[keep] if control[p] else 0)
    final_values=[decode(aes(seed,range(3+2*blocks,3+3*blocks))) for seed in seeds]
    sign=-1 if control[1] else 1
    final=[(sign*(final_values[1][c]-final_values[0][c]-along[c]))&mask for c in range(dim)]
    suffix=b''.join(corrections)+b''.join(v.to_bytes(8,'little') for v in final)
    return [roots[:16]+suffix,roots[16:]+suffix]


def main():
    rng=np.random.default_rng(2026102925);rows=[]
    for dim in (1,2,4,10,52):
        bits=8;n=32;alpha=rng.integers(0,256,n,dtype=np.uint64);alpha[:4]=[0,1,254,255]
        beta=random64((n,dim));root_bytes=os.urandom(n*32);roots=np.frombuffer(root_bytes,dtype=np.uint8)
        keys=generate(alpha,beta,bits,roots)
        refs=[reference(int(a),b,bits,root_bytes[i*32:(i+1)*32]) for i,(a,b) in enumerate(zip(alpha,beta))]
        for p in (0,1):assert keys[p].tobytes()==b''.join(v[p] for v in refs)
        if dim<=2:
            old=small_gen(alpha,beta,bits,64,roots)
            assert all(np.array_equal(a,b) for a,b in zip(keys,old))
        for x in (alpha,(alpha-1)&np.uint64(255),(alpha+1)&np.uint64(255),rng.integers(0,256,n,dtype=np.uint64)):
            values=[select(keys[p],x,np.arange(dim),p,bits,dim) for p in (0,1)]
            assert np.array_equal(values[0]+values[1],np.where((x<alpha)[:,None],beta,0))
        rows.append(dict(bits=bits,dim=dim,independent_reference_keys=2*n,all_passed=True))
    bits=63;n=16;dim=52;alpha=rng.integers(0,1<<bits,n,dtype=np.uint64)
    alpha[:6]=[0,1,(1<<62)-1,1<<62,(1<<63)-2,(1<<63)-1]
    beta=random64((n,dim));root_bytes=os.urandom(n*32);keys=generate(alpha,beta,bits,np.frombuffer(root_bytes,dtype=np.uint8))
    for p in (0,1):assert keys[p].tobytes()==b''.join(reference(int(a),b,bits,root_bytes[i*32:(i+1)*32])[p] for i,(a,b) in enumerate(zip(alpha,beta)))
    for x in (alpha,(alpha-1)&np.uint64((1<<63)-1),(alpha+1)&np.uint64((1<<63)-1)):
        values=[select(keys[p],x,np.arange(dim),p,bits,dim) for p in (0,1)]
        assert np.array_equal(values[0]+values[1],np.where((x<alpha)[:,None],beta,0))
    rows.append(dict(bits=bits,dim=dim,independent_reference_keys=2*n,all_passed=True))
    pieces=[(0,17,2,3),(17,64,-1,47),(64,128,0,29),(128,193,0,-41),(193,256,3,-769)]
    spec=dict(unsigned_pieces=pieces);alpha=np.arange(256,dtype=np.uint64)
    f=np.asarray([next(m*x+c for a,b,m,c in pieces if a<=x<b) for x in range(256)],dtype=np.int64)
    all_keys={v:prepare(256,spec,8,alpha,variant=v)[0] for v in ('basis','wide')}
    for h in range(256):
        opened=np.full(256,h,dtype=np.uint64)
        for variant in ('basis','wide'):
            local=[]
            for cached in (0,1,2,3):
                values=[evaluate(all_keys[variant][p],opened,spec,p,8,variant=variant,cached=cached) for p in (0,1)]
                assert np.array_equal((values[0]+values[1])&np.uint64(255),f[(h-alpha.astype(np.int64))%256].view(np.uint64)&np.uint64(255))
                if local:
                    for p in (0,1):assert np.array_equal(values[p],local[p])
                local=values
    out=dict(primitive_checks=rows,spline_mask_input_pairs=65536,spline_variants=['basis','wide'],
        native_modes_per_variant=4,all_passed=True,native_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        limits='Correctness and independent AES-provider serialization only. Standard DCF with extended public payload dimension. No concrete implementation security/constant-time or novelty claim.')
    (ROOT/'results/dcf_wide_native_check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)


if __name__=='__main__':main()
