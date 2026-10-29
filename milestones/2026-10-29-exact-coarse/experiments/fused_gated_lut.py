"""Dense two-payload offset tables for SiLU(a)*b versus table + Beaver.

Dense, reference DPF or native DPF backends; trusted parent, two local workers.
No complete private LLM. Local
stochastic truncation consumes eight ring bits; this is not same-ring truncation.
All masks/tables are fresh per evaluation, even when public SiLU table is reused.
Local stochastic rounding below is a NUMERICAL/FUNCTIONAL legacy experiment:
its output privacy is not established under standard2PC. Use the faithful FSS
postprocessing in gated_lut_strong_baselines.py for the current candidate.
"""
import json
import argparse
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from dpf_reference import generate,encode,decode,evaluate_all
from dpf_aes_native import generate_batch,lookup_batch

ROOT=Path(__file__).resolve().parents[1]


def random64(shape):
    return np.frombuffer(os.urandom(int(np.prod(shape))*8),dtype='<u8').copy().reshape(shape)


def split64(x,ring_bits=64):
    mask=np.uint64((1<<ring_bits)-1)
    a=random64(x.shape)&mask
    dtype={64:np.uint64,32:np.uint32,16:np.uint16}[ring_bits]
    return [a.astype(dtype),(np.subtract(x,a,dtype=np.uint64)&mask).astype(dtype)]


def split8(x):
    a=np.frombuffer(os.urandom(x.size),dtype=np.uint8).copy()
    return [a,np.subtract(x,a,dtype=np.uint8)]


def public_table():
    code=np.arange(256,dtype=np.int64)
    signed=(code^128)-128
    real=signed/16.0
    return np.rint(real*expit(real)*256).astype(np.int64).view(np.uint64)


def prepare(a,b,mode,backend='dense',table_values=None,ring_bits=64,rounding=True):
    start=time.perf_counter_ns();n=len(a)
    mask=np.uint64((1<<ring_bits)-1)
    ra=np.frombuffer(os.urandom(n),dtype=np.uint8).copy()
    rb=random64((n,))&mask;ras=split8(ra);rbs=split64(rb,ring_bits)
    inputs=list(zip(split8(a),split64(b,ring_bits)))
    if backend in ('compact','native'):
        if backend=='native':
            payload=np.ones((n,2 if mode=='fused' else 1),dtype=np.uint64)
            if mode=='fused':payload[:,1]=np.negative(rb,dtype=np.uint64)
            dpfs=generate_batch(ra,payload,group_bits=ring_bits)
        else:
            dpfs=[[],[]]
            for j in range(n):
                payload=np.array([1,(-int(rb[j]))%(1<<64)] if mode=='fused' else [1],dtype=np.uint64)
                generated=generate(int(ra[j]),payload)
                for i in (0,1):dpfs[i].append(encode(generated[i]))
        keys=[dict(ra=ras[i],rb=rbs[i],dpf=dpfs[i]) for i in (0,1)]
        if mode=='baseline':
            rv=random64((n,))&mask;rvs=split64(rv,ring_bits)
            products=split64(np.multiply(rv,rb,dtype=np.uint64),ring_bits)
            for i in (0,1):keys[i].update(rv=rvs[i],product=products[i])
    else:
        index=np.subtract(np.arange(256,dtype=np.uint8)[None,:],ra[:,None],dtype=np.uint8)
        values=public_table() if table_values is None else table_values
        function=values[index] if values.ndim==1 else values[np.arange(n)[:,None],index]
    if backend=='dense' and mode=='fused':
        # F0(t)=g(t-ra), F1(t)=-rb*g(t-ra), all in Z/(2^64).
        intercept=np.negative(np.multiply(rb[:,None],function,dtype=np.uint64),dtype=np.uint64)
        table=np.stack([function,intercept],axis=2)
        ts=split64(table,ring_bits)
        keys=[dict(ra=ras[i],rb=rbs[i],table=ts[i]) for i in (0,1)]
    elif backend=='dense':
        rv=random64((n,))&mask;rvs=split64(rv,ring_bits)
        ts=split64(np.add(function,rv[:,None],dtype=np.uint64),ring_bits)
        products=split64(np.multiply(rv,rb,dtype=np.uint64),ring_bits)
        keys=[dict(ra=ras[i],rb=rbs[i],rv=rvs[i],product=products[i],table=ts[i]) for i in (0,1)]
    # Actual array sizes supplied to workers, not theoretical compressed keys.
    # Re-randomize output shares with a fresh zero sharing before rounding. This
    # gives exact uniform low-share bits even for the computational DPF backend.
    if rounding:
        zeros=split64(np.zeros(n,dtype=np.uint64),ring_bits)
        for i in (0,1):keys[i]['rounding_zero']=zeros[i]
    material=sum(sum(map(len,v)) if isinstance(v,list) else v.nbytes for key in keys for v in key.values())
    return inputs,keys,{'dealer_ms':(time.perf_counter_ns()-start)/1e6,'key_array_bytes_both_parties':material}


def receive(sock,length):
    buf=bytearray(length);view=memoryview(buf);done=0
    while done<length:
        amount=sock.recv_into(view[done:])
        if amount==0:raise RuntimeError('peer closed')
        done+=amount
    return buf


def exchange(sock,rank,packet):
    if rank==0:
        sock.sendall(packet);other=receive(sock,len(packet))
    else:
        other=receive(sock,len(packet));sock.sendall(packet)
    return other


def worker(rank,sock,inputs,key,mode,backend,table_values,ring_bits,barrier,pipe):
    try:
        a,b=inputs;n=len(a)
        mask=np.uint64((1<<ring_bits)-1);wire_dtype='<u8' if ring_bits==64 else '<u4'
        local_a=np.add(a,key['ra'],dtype=np.uint8)
        local_b=np.add(b,key['rb'],dtype=np.uint64)&mask
        packet=local_a.tobytes()+local_b.astype(wire_dtype).tobytes()
        barrier.wait()
        start=time.perf_counter_ns()
        other=exchange(sock,rank,packet)
        ahat=np.add(local_a,np.frombuffer(other,dtype=np.uint8,count=n),dtype=np.uint8)
        bhat=np.add(local_b,np.frombuffer(other,dtype=wire_dtype,count=n,offset=n),dtype=np.uint64)&mask
        row=np.arange(n)
        if backend=='compact':
            dim=2 if mode=='fused' else 1
            evaluated=np.empty((n,dim),dtype=np.uint64)
            table=table_values;domain=np.arange(256,dtype=np.uint8)
            for j in range(n):
                dpf=evaluate_all(decode(key['dpf'][j],rank,8,dim))
                local_table=table if table.ndim==1 else table[j]
                weights=local_table[np.subtract(ahat[j],domain,dtype=np.uint8)]
                evaluated[j]=np.multiply(dpf,weights[:,None],dtype=np.uint64).sum(axis=0,dtype=np.uint64)
        elif backend=='native':
            evaluated=lookup_batch(key['dpf'],ahat,table_values,rank,2 if mode=='fused' else 1,group_bits=ring_bits)
        if mode=='fused':
            coeff=evaluated if backend in ('compact','native') else key['table'][row,ahat,:]
            result=np.add(np.multiply(bhat,coeff[:,0],dtype=np.uint64),coeff[:,1],dtype=np.uint64)
            sent=len(packet);exchanges=1
        else:
            vshare=np.add(evaluated[:,0],key['rv'],dtype=np.uint64) if backend in ('compact','native') else key['table'][row,ahat]
            vpacket=(vshare&mask).astype(wire_dtype).tobytes()
            vpeer=exchange(sock,rank,vpacket)
            vhat=np.add(vshare,np.frombuffer(vpeer,dtype=wire_dtype),dtype=np.uint64)&mask
            result=np.subtract(key['product'],np.multiply(vhat,key['rb'],dtype=np.uint64),dtype=np.uint64)
            result=np.subtract(result,np.multiply(bhat,key['rv'],dtype=np.uint64),dtype=np.uint64)
            if rank==0:result=np.add(result,np.multiply(vhat,bhat,dtype=np.uint64),dtype=np.uint64)
            sent=len(packet)+len(vpacket);exchanges=2
        result=result&mask
        # Standard local stochastic truncation into a SMALLER ring. Adding S-1
        # makes floor(a0/S)+floor(a1/S) unbiased rather than downward biased.
        rounding_share=np.add(result,key['rounding_zero'],dtype=np.uint64)&mask
        if rank==0:rounding_share=np.add(rounding_share,np.uint64(255),dtype=np.uint64)&mask
        rounding_share=rounding_share>>np.uint64(8)
        end=time.perf_counter_ns()
        # Returning shares to the test parent is outside measured online scope.
        pipe.send({'rank':rank,'online_ms':(end-start)/1e6,'sent_bytes':sent,'received_bytes':sent,
                   'logical_exchanges':exchanges,'share':result,'rounded_share':rounding_share})
    except BaseException as err:
        pipe.send({'rank':rank,'error':repr(err)})
    finally:
        pipe.close();sock.close()


def run(a,b,mode,backend='dense',table_values=None,return_values=False,ring_bits=64):
    assert ring_bits in (32,64) and (ring_bits==64 or backend in ('native','dense'))
    mask=np.uint64((1<<ring_bits)-1);outmask=np.uint64((1<<(ring_bits-8))-1)
    values=public_table() if table_values is None else np.ascontiguousarray(table_values,dtype=np.uint64)
    assert values.shape in ((256,),(len(a),256))
    inputs,keys,offline=prepare(a,b,mode,backend,values,ring_bits)
    # fork only avoids huge benchmark setup pickles. This trusted test harness
    # is not a process-isolated deployment: parent memory contains both keys.
    ctx=mp.get_context('fork');sockets=socket.socketpair();barrier=ctx.Barrier(2)
    parents=[];processes=[]
    for rank in (0,1):
        parent,child=ctx.Pipe(duplex=False)
        proc=ctx.Process(target=worker,args=(rank,sockets[rank],inputs[rank],keys[rank],mode,backend,values,ring_bits,barrier,child))
        proc.start();child.close();parents.append(parent);processes.append(proc)
    for sock in sockets:sock.close()
    results=[]
    for p in parents:
        if not p.poll(30):
            for proc in processes:proc.terminate()
            raise RuntimeError('worker result timeout')
        results.append(p.recv())
    for p in processes:
        p.join(30)
        if p.is_alive():p.terminate();raise RuntimeError('worker timeout')
        assert p.exitcode==0
    if any('error' in r for r in results):raise RuntimeError(results)
    actual=np.add(results[0].pop('share'),results[1].pop('share'),dtype=np.uint64)&mask
    activation=values[a] if values.ndim==1 else values[np.arange(len(a)),a]
    expected=np.multiply(activation,b,dtype=np.uint64)&mask
    assert np.array_equal(actual,expected)
    rounded=np.add(results[0].pop('rounded_share'),results[1].pop('rounded_share'),dtype=np.uint64)&outmask
    floor=expected>>np.uint64(8)
    ceil=(floor+np.uint64(1))&outmask
    assert np.all((rounded==floor)|(rounded==ceil))
    assert sum(r['sent_bytes'] for r in results)==len(a)*(2+(ring_bits//8)*(2 if mode=='fused' else 4))
    result=dict(mode=mode,backend=backend,ring_bits=ring_bits,n=len(a),correct=True,rounded_output_in_floor_ceil=True,offline=offline,parties=results,
                total_online_sent_bytes=sum(r['sent_bytes'] for r in results),public_table_bytes=values.nbytes)
    if return_values:result.update(_raw_values=actual,_rounded_values=rounded)
    return result


def exhaustive_rounding_check():
    # All secrets and all first shares in Z/256, reduced to Z/32.
    first=np.arange(256,dtype=np.uint16);checks=0
    for secret in range(256):
        second=(secret-first)&255
        out=((((first+7)&255)>>3)+(second>>3))&31
        counts=np.bincount(out,minlength=32)
        target=np.zeros(32,dtype=np.int64);q,r=divmod(secret,8)
        target[q]+=(8-r)*32;target[(q+1)&31]+=r*32
        assert np.array_equal(counts,target)
        checks+=256
    return {'ring_bits':8,'fractional_removed':3,'secret_share_pairs':checks,'exact_stochastic_law_checked':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('dense','compact','native'),default='dense')
    parser.add_argument('--sizes',default='128,1024,4096,14336')
    parser.add_argument('--ring-bits',type=int,choices=(32,64),default=64)
    args=parser.parse_args();sizes=[int(v) for v in args.sizes.split(',')]
    rng=np.random.default_rng(291501)
    # Entire 8-bit gate domain crossed with signed and ring-valued b boundaries.
    boundaries=np.array([-(1<<63),-(1<<31),-32768,-1,0,1,32767,(1<<31)-1,(1<<63)-1],dtype=np.int64).view(np.uint64)
    a=np.repeat(np.arange(256,dtype=np.uint8),len(boundaries));b=np.tile(boundaries,256)
    checks=[run(a,b,mode,args.backend,ring_bits=args.ring_bits) for mode in ('baseline','fused')]
    rows=[]
    for n in sizes:
        for repetition in range(3):
            a=rng.integers(0,256,n,dtype=np.uint8)
            b=rng.integers(-32768,32768,n,dtype=np.int64).view(np.uint64)
            order=('baseline','fused') if repetition%2==0 else ('fused','baseline')
            for mode in order:
                record=run(a,b,mode,args.backend,ring_bits=args.ring_bits);record['repetition']=repetition;record['order']=list(order);rows.append(record)
        print('finished',n,flush=True)
    out={'description':'Coefficient/intercept function shares for one-exchange g(a)*b versus scalar LUT and Beaver with b opening batched into first exchange',
         'backend':args.backend,
         'gate_index_bits':8,'gate_input_scale':16,'lookup_output_scale':256,'second_input_scale':256,
         'raw_output_scale':65536,'ring':f'Z/(2^{args.ring_bits})','rounded_output_scale':256,'rounded_output_ring_bits':args.ring_bits-8,
         'rounding_check':exhaustive_rounding_check(),'boundary_checks':checks,'measurements':rows,
         'security_scope':'Trusted-dealer fresh per-element masks. Dense backend uses full additive function tables; compact/native use standard vector-payload seed-tree DPF with SHAKE256/AES-128 and 128-bit roots, computational PRG assumption, not a concrete audited security bound. Local fork harness knows/inherits all keys, not a secure deployment.',
         'limits':'Legacy local stochastic rounding is functional only, NOT established as secure2PC truncation. No real model, input quantization, SAME-RING truncation/lifting, dealer network, WAN or model projections. Rounding reduces the ring and cannot be treated as free same-ring output. Use gated_lut_strong_baselines.py faithful mode for the current candidate. Exact raw functionality is finite LUT g(a) times ring b, not continuous SiLU.',
         'novelty':'Not established. General offset FSS, private LUTs with vector payloads and function-dependent preprocessing are prior art. Need compare exact construction and compact instantiations before claiming contribution.'}
    filename='fused_gated_lut.json' if args.backend=='dense' else f'fused_gated_lut_{args.backend}.json'
    if args.ring_bits!=64:filename=filename.removesuffix('.json')+str(args.ring_bits)+'.json'
    (ROOT/'results'/filename).write_text(json.dumps(out,indent=2)+'\n')
    for n in sizes:
        for mode in ('baseline','fused'):
            rr=[r for r in rows if r['n']==n and r['mode']==mode]
            print(n,mode,'online_ms',np.median([r['parties'][0]['online_ms'] for r in rr]),
                  'dealer_ms',np.median([r['offline']['dealer_ms'] for r in rr]),'online_bytes',rr[0]['total_online_sent_bytes'],
                  'key_bytes',rr[0]['offline']['key_array_bytes_both_parties'])


if __name__=='__main__':main()
