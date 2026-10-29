"""Cross-check native AES DPF against Python tree + independent AES provider."""
import hashlib
import json
import os
import struct
import time

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes

import dpf_reference as ref
from dpf_aes_native import ROOT,LIB,generate_batch,lookup_batch


def aes_blocks(seed,counters):
    block=b''.join(struct.pack('<QQ',c,0x4c4c4d4450467631) for c in counters)
    cipher=Cipher(algorithms.AES(seed),modes.ECB()).encryptor()
    return cipher.update(block)+cipher.finalize()


def expand(seed):
    blocks=aes_blocks(seed,(0,1,2));t=blocks[32]
    return (blocks[:16],t&1),(blocks[16:32],(t>>1)&1)


def convert(seed,dim,group_bits=64):
    return np.frombuffer(aes_blocks(seed,(3,)),count=dim,dtype=f'<u{group_bits//8}').astype(np.uint64)


def encoded_group(key,group_bits):
    if group_bits==64:return ref.encode(key)
    header=key['root']+b''.join(s+bytes([tl|(tr<<1)]) for s,tl,tr in key['corrections'])
    return header+key['final'].astype(f'<u{group_bits//8}').tobytes()


def main():
    ref.expand=expand
    rows=[];rng=np.random.default_rng(291601)
    for group_bits,dim in ((64,1),(64,2),(32,1),(32,2),(16,1),(16,2)):
        mask=np.uint64((1<<group_bits)-1)
        ref.convert=lambda seed,dim:convert(seed,dim,group_bits)
        alpha=np.arange(256,dtype=np.uint8)
        beta=np.frombuffer(os.urandom(256*dim*8),dtype='<u8').copy().reshape(256,dim)&mask
        expected=[ref.generate(int(a),b) for a,b in zip(alpha,beta)]
        roots=np.frombuffer(b''.join(k[0]['root']+k[1]['root'] for k in expected),dtype=np.uint8)
        keys=generate_batch(alpha,beta,roots,group_bits=group_bits)
        for party in (0,1):
            want=np.frombuffer(b''.join(encoded_group(k[party],group_bits) for k in expected),dtype=np.uint8).reshape(keys[party].shape)
            assert np.array_equal(keys[party],want)
        table=np.frombuffer(os.urandom(256*8),dtype='<u8').copy()
        # Native LUT for every masked input, each hidden point appears once.
        for ahat_int in range(256):
            ahat=np.full(256,ahat_int,dtype=np.uint8)
            out=[lookup_batch(keys[p],ahat,table,p,dim,group_bits=group_bits) for p in (0,1)]
            actual=np.add(*out,dtype=np.uint64)&mask
            wanted=np.multiply(table[np.subtract(ahat,alpha,dtype=np.uint8)][:,None],beta,dtype=np.uint64)&mask
            assert np.array_equal(actual,wanted)
        per_row=np.frombuffer(os.urandom(256*256*8),dtype='<u8').copy().reshape(256,256)
        ahat=rng.integers(0,256,256,dtype=np.uint8)
        per_output=[lookup_batch(keys[p],ahat,per_row,p,dim,group_bits=group_bits) for p in (0,1)]
        expected_rows=np.multiply(per_row[np.arange(256),np.subtract(ahat,alpha,dtype=np.uint8)][:,None],beta,dtype=np.uint64)&mask
        assert np.array_equal(np.add(*per_output,dtype=np.uint64)&mask,expected_rows)
        rows.append({'group_bits':group_bits,'payload_dimension':dim,'key_serializations_against_python_and_independent_aes':512,
                     'hidden_point_masked_input_pairs':65536,'distinct_public_table_rows_checked':256,'passed':True})
    result={'checks':rows,'native_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
            'compiler':'c++ Ubuntu 13.3.0, -O3 -std=c++17 -maes -msse2 -fPIC -shared',
            'prg':'AES-128 keyed by seed, distinct 128-bit counter blocks 0/1/2 for tree and 3 for leaf; fresh OS128 roots',
            'limits':'Functional/cross-provider evidence only; does not prove PRG security, concrete security losses or implementation isolation/constant-time behavior.'}
    (ROOT/'results/dpf_aes_native_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
