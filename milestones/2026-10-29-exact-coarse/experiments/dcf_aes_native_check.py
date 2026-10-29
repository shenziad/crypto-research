"""DCF checks against Python correction algebra and independent AES provider.

Standard algorithm: Microsoft Research EzPC/FSS/src/dcf.cpp (MIT); original
source and full license saved at sources/llama_reference/. Not new crypto.
"""
import hashlib
import json
import os
import struct

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
from dcf_aes_native import generate_batch,evaluate_batch
from dpf_aes_native import ROOT,LIB


def aes(seed,counters):
    cipher=Cipher(algorithms.AES(seed),modes.ECB()).encryptor()
    data=b''.join(struct.pack('<QQ',c,0x4c4c4d4443467631) for c in counters)
    return cipher.update(data)+cipher.finalize()


def words(block,dim,bits):
    size=bits//8
    return [int.from_bytes(block[j*size:(j+1)*size],'little') for j in range(dim)]


def generate_reference(alpha,beta,bits,group_bits,roots):
    dim=len(beta);mask=(1<<group_bits)-1;size=group_bits//8
    seeds=[roots[:16],roots[16:]];control=[0,1];along=[0]*dim;corrections=[]
    for level in range(bits):
        raw=[aes(s,(0,1,2,3,4)) for s in seeds]
        child=[[v[:16],v[16:32]] for v in raw]
        t=[[(v[32]&1),((v[32]>>1)&1)] for v in raw]
        values=[[words(v[48:64],dim,group_bits),words(v[64:80],dim,group_bits)] for v in raw]
        keep=(alpha>>(bits-1-level))&1;lose=keep^1;sign=-1 if control[1] else 1
        vcw=[(sign*(-along[c]-values[0][lose][c]+values[1][lose][c]+(int(beta[c]) if keep else 0)))&mask for c in range(dim)]
        along=[(along[c]-values[1][keep][c]+values[0][keep][c]+sign*vcw[c])&mask for c in range(dim)]
        scw=bytes(a^b for a,b in zip(child[0][lose],child[1][lose]))
        tcw=[t[0][0]^t[1][0]^keep^1,t[0][1]^t[1][1]^keep]
        corrections.append(scw+bytes([tcw[0]|(tcw[1]<<1)])+b''.join(v.to_bytes(size,'little') for v in vcw))
        for party in (0,1):
            seeds[party]=bytes(a^b for a,b in zip(child[party][keep],scw)) if control[party] else child[party][keep]
            control[party]=t[party][keep]^(tcw[keep] if control[party] else 0)
    f0=words(aes(seeds[0],(5,)),dim,group_bits);f1=words(aes(seeds[1],(5,)),dim,group_bits)
    sign=-1 if control[1] else 1
    final=[(sign*(f1[c]-f0[c]-along[c]))&mask for c in range(dim)]
    suffix=b''.join(corrections)+b''.join(v.to_bytes(size,'little') for v in final)
    return [roots[:16]+suffix,roots[16:]+suffix]


def main():
    rows=[];rng=np.random.default_rng(291701)
    for group_bits,dim in ((16,1),(16,2),(32,1),(32,2),(64,1),(64,2)):
        mask=np.uint64((1<<group_bits)-1);a=np.arange(256,dtype=np.uint64)
        beta=np.frombuffer(os.urandom(256*dim*8),dtype='<u8').copy().reshape(256,dim)&mask
        root_bytes=os.urandom(256*32);roots=np.frombuffer(root_bytes,dtype=np.uint8)
        keys=generate_batch(a,beta,8,group_bits,roots)
        reference=[generate_reference(int(v),b,8,group_bits,root_bytes[i*32:(i+1)*32]) for i,(v,b) in enumerate(zip(a,beta))]
        for p in (0,1):assert keys[p].tobytes()==b''.join(k[p] for k in reference)
        allx=np.tile(np.arange(256,dtype=np.uint64),256)
        repeated=[np.repeat(k,256,axis=0) for k in keys]
        z=[evaluate_batch(repeated[p],allx,p,8,dim,group_bits) for p in (0,1)]
        target=np.where((allx<np.repeat(a,256))[:,None],np.repeat(beta,256,axis=0),0)
        assert np.array_equal((z[0]+z[1])&mask,target)
        rows.append({'input_bits':8,'group_bits':group_bits,'payload_dimension':dim,
                     'independent_serialized_keys_checked':512,'threshold_input_pairs':65536,'passed':True})
    for bits in (16,24,32,56):
        n=4096;mask=np.uint64((1<<bits)-1)
        a=rng.integers(0,1<<bits,n,dtype=np.uint64)
        a[:6]=[0,1,(1<<(bits-1))-1,1<<(bits-1),(1<<bits)-2,(1<<bits)-1]
        b=np.frombuffer(os.urandom(n*8),dtype='<u8').copy()[:,None]
        keys=generate_batch(a,b,bits,64)
        for x in (a,(a-1)&mask,(a+1)&mask,rng.integers(0,1<<bits,n,dtype=np.uint64)):
            z=[evaluate_batch(keys[p],x,p,bits,1,64) for p in (0,1)]
            assert np.array_equal(z[0]+z[1],np.where((x<a)[:,None],b,0))
        rows.append({'input_bits':bits,'group_bits':64,'payload_dimension':1,'boundary_random_pairs':n*4,'passed':True})
    out={'checks':rows,'native_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
         'upstream_dcf_blob_sha':'e43f3d1753296ee80f3bd4c93388d272cbcfc0ce',
         'limits':'Functional and independent AES-provider/key-algebra checks only; standard DCF construction, not security or constant-time validation.'}
    (ROOT/'results/dcf_aes_native_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
