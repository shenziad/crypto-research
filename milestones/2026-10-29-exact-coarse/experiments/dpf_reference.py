"""Research reference for the standard two-party seed-tree DPF construction.

BGI tree/correction-word structure, see https://eprint.iacr.org/2018/707 .
Fresh 128-bit roots; domain-separated SHAKE256 used as the PRG/hash instantiation.
Variable-time Python, no implementation-security audit or concrete security bound.
Group is (Z/(2^64))^dimension. Vector payloads are standard DPF functionality.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def xor(a,b):
    return bytes(x^y for x,y in zip(a,b))


def expand(seed):
    raw=hashlib.shake_256(b'LLMCRYPTO/DPF/tree/v1\x00'+seed).digest(34)
    return (raw[:16],raw[16]&1),(raw[17:33],raw[33]&1)


def convert(seed,dim):
    raw=hashlib.shake_256(b'LLMCRYPTO/DPF/leaf/v1\x00'+seed).digest(8*dim)
    return np.frombuffer(raw,dtype='<u8').copy()


def generate(alpha,beta,bits=8):
    beta=np.asarray(beta,dtype=np.uint64)
    roots=[os.urandom(16),os.urandom(16)]
    seeds=roots.copy();controls=[0,1];corrections=[]
    for level in range(bits):
        keep=(alpha>>(bits-1-level))&1;lose=keep^1
        children=[expand(seed) for seed in seeds]
        scw=xor(children[0][lose][0],children[1][lose][0])
        tcw=(children[0][0][1]^children[1][0][1]^keep^1,
             children[0][1][1]^children[1][1][1]^keep)
        corrections.append((scw,*tcw))
        for b in (0,1):
            seed,t=children[b][keep]
            if controls[b]:seed=xor(seed,scw);t^=tcw[keep]
            seeds[b]=seed;controls[b]=t
    final=np.subtract(beta,convert(seeds[0],len(beta)),dtype=np.uint64)
    final=np.add(final,convert(seeds[1],len(beta)),dtype=np.uint64)
    if controls[1]:final=np.negative(final,dtype=np.uint64)
    return [dict(party=b,root=roots[b],corrections=corrections,final=final.copy()) for b in (0,1)]


def encode(key):
    return key['root']+b''.join(seed+bytes([tl|(tr<<1)]) for seed,tl,tr in key['corrections'])+key['final'].astype('<u8').tobytes()


def decode(data,party,bits,dim):
    assert len(data)==16+17*bits+8*dim
    corrections=[]
    for level in range(bits):
        off=16+17*level
        corrections.append((data[off:off+16],data[off+16]&1,(data[off+16]>>1)&1))
    return dict(party=party,root=data[:16],corrections=corrections,
                final=np.frombuffer(data,dtype='<u8',count=dim,offset=16+17*bits).copy())


def evaluate_all(key):
    states=[(key['root'],key['party'])]
    for scw,tl,tr in key['corrections']:
        new=[]
        for seed,control in states:
            children=expand(seed)
            for branch,tcw in ((0,tl),(1,tr)):
                s,t=children[branch]
                if control:s=xor(s,scw);t^=tcw
                new.append((s,t))
        states=new
    dim=len(key['final']);out=np.empty((len(states),dim),dtype=np.uint64)
    for i,(seed,control) in enumerate(states):
        out[i]=convert(seed,dim)
        if control:out[i]=np.add(out[i],key['final'],dtype=np.uint64)
    if key['party']:out=np.negative(out,dtype=np.uint64)
    return out


def check():
    checked=0
    for alpha in range(256):
        payload=np.array([1,int.from_bytes(os.urandom(8),'little')],dtype=np.uint64)
        keys=generate(alpha,payload)
        keys=[decode(encode(k),b,8,2) for b,k in enumerate(keys)]
        y=np.add(evaluate_all(keys[0]),evaluate_all(keys[1]),dtype=np.uint64)
        target=np.zeros((256,2),dtype=np.uint64);target[alpha]=payload
        assert np.array_equal(y,target)
        checked+=len(y)
    result={'kind':'functional DPF check, not an empirical security test','points':256,
            'domain_points_per_key':256,'output_coordinates':2,'vector_evaluations_checked':checked,
            'seed_bits':128,'key_bytes_scalar':16+17*8+8,'key_bytes_vector2':16+17*8+16,
            'backend':'domain-separated SHAKE256 seed tree, Python reference; not constant-time/audited',
            'passed':True}
    path=Path(__file__).resolve().parents[1]/'results/dpf_reference_check.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':check()
