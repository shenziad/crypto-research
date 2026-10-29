"""Matched finite gated operator ending in signed64 additive output shares.

Baselines: LUT+Beaver32, LUT16+LLAMA signed16x16 multiplication. Candidate:
two-payload DPF in ring32. Default uses faithful FSS floor32/256 -> signed64.
Legacy local stochastic32->24 is retained ONLY as an unproved functional mode.
Only local processes; optional per-exchange delay is explicitly emulated RTT.
Standard LLAMA algebra is prior art, not a contribution of this prototype.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time

import numpy as np
from fused_gated_lut import ROOT,prepare,public_table,random64,split64,split8,exchange
from dpf_aes_native import generate_batch as dpf_gen,lookup_batch,LIB
from dcf_aes_native import generate_batch as dcf_gen,evaluate_batch as dcf_eval

M16=np.uint64(65535);M24=np.uint64((1<<24)-1);M32=np.uint64((1<<32)-1)


def material_bytes(keys):
    return sum(v.nbytes for k in keys for v in k.values())


def prepare_lift(n):
    r=random64((n,))&M24
    masks=[v&M24 for v in split64(r,32)]
    constants=split64(np.negative(r+np.uint64(1<<23),dtype=np.uint64))
    dcf=dcf_gen(r,np.full((n,1),1<<24,dtype=np.uint64),24,64)
    return [dict(lift_mask=masks[i],lift_const=constants[i],lift_dcf=dcf[i]) for i in (0,1)]


def prepare_mixed(a,b,rounding=True):
    n=len(a);ra=np.frombuffer(os.urandom(n),dtype=np.uint8).copy()
    rb=random64((n,))&M16;rg=random64((n,))&M16
    ras=split8(ra);rbs=split64(rb,32);rgs=split64(rg,32);products=split64(rb*rg,32)
    dpf=dpf_gen(ra,np.ones((n,1),dtype=np.uint64),group_bits=16)
    bx=np.stack([np.ones(n,dtype=np.uint64),np.negative(rb,dtype=np.uint64)&M16],axis=1)
    by=np.stack([np.ones(n,dtype=np.uint64),np.negative(rg,dtype=np.uint64)&M16],axis=1)
    dx=dcf_gen(rg,bx,16,16);dy=dcf_gen(rb,by,16,16)
    keys=[dict(ra=ras[i],rb=rbs[i],rg=rgs[i],product=products[i],dpf=dpf[i],dcfx=dx[i],dcfy=dy[i]) for i in (0,1)]
    if rounding:
        zeros=split64(np.zeros(n,dtype=np.uint64),32)
        for i in (0,1):keys[i]['rounding_zero']=zeros[i]
    # Common start: b is shared in ring32, with signed16 value promised by the
    # quantizer. Mixed multiplication projects these shares mod2^16 locally.
    inputs=list(zip(split8(a),split64(b,32)))
    return inputs,keys


def prepare_faithful(n):
    # Existing FSS right-shift identity, merged with signed extension on the
    # SAME opened32 mask. DCFs are evaluated on low8 and full32 public views.
    r=random64((n,))&M32;masks=split64(r,32)
    constant=np.negative((r>>np.uint64(8))+np.where(r<np.uint64(1<<31),np.uint64(1<<24),np.uint64(0)),dtype=np.uint64)
    constants=split64(constant)
    low=dcf_gen(r&np.uint64(255),np.full((n,1),np.uint64((1<<64)-1),dtype=np.uint64),8,64)
    high=dcf_gen((r+np.uint64(1<<31))&M32,np.full((n,1),1<<24,dtype=np.uint64),32,64)
    return [dict(ars_mask=masks[i],ars_const=constants[i],ars_low=low[i],ars_high=high[i]) for i in (0,1)]


def pack24(x):
    return np.ascontiguousarray(x,dtype='<u4').view(np.uint8).reshape(-1,4)[:,:3].copy().tobytes()


def unpack24(buf):
    x=np.frombuffer(buf,dtype=np.uint8).reshape(-1,3).astype(np.uint64)
    return x[:,0]|(x[:,1]<<8)|(x[:,2]<<16)


class Channel:
    def __init__(self,sock,party,rtt_ms):
        self.sock=sock;self.party=party;self.rtt_ms=rtt_ms;self.sent=0;self.exchanges=0
    def swap(self,packet):
        other=exchange(self.sock,self.party,packet)
        self.sent+=len(packet);self.exchanges+=1
        if self.rtt_ms:time.sleep(self.rtt_ms/1000)
        return other


def worker(party,sock,inputs,key,mode,table,rtt_ms,postprocess,barrier,pipe):
    try:
        a,b=inputs;n=len(a);ch=Channel(sock,party,rtt_ms)
        prefix_bytes=0;prefix_exchanges=0;prefix_ms=0.0;quantized=None
        if 'q_mask' in key:
            from gated_input_quantizer import online as quantize_online
            barrier.wait();start=time.perf_counter_ns()
            quantized=quantize_online(party,ch,np.concatenate([a,b]),key)
            a=(quantized[:n]&np.uint64(255)).astype(np.uint8);b=quantized[n:]
            prefix_bytes=ch.sent;prefix_exchanges=ch.exchanges
            after_prefix=time.perf_counter_ns();prefix_ms=(after_prefix-start)/1e6
        width=2 if mode=='mixed16' else 4;wire='<u2' if width==2 else '<u4'
        local_a=np.add(a,key['ra'],dtype=np.uint8)
        local_b=(np.add(b,key['rb'],dtype=np.uint64)&(M16 if width==2 else M32))
        packet=local_a.tobytes()+local_b.astype(wire).tobytes()
        if 'q_mask' not in key:
            barrier.wait();start=time.perf_counter_ns();after_prefix=start
        peer=ch.swap(packet)
        ahat=np.add(local_a,np.frombuffer(peer,dtype=np.uint8,count=n),dtype=np.uint8)
        bhat=(local_b+np.frombuffer(peer,dtype=wire,count=n,offset=n))&(M16 if width==2 else M32)
        dim=2 if mode=='fused32' else 1
        values=lookup_batch(key['dpf'],ahat,table,party,dim,group_bits=16 if mode=='mixed16' else 32)
        if mode=='fused32':raw=(bhat*values[:,0]+values[:,1])&M32
        elif mode=='beaver32':
            local_g=(values[:,0]+key['rv'])&M32
            peer=ch.swap(local_g.astype('<u4').tobytes())
            ghat=(local_g+np.frombuffer(peer,dtype='<u4'))&M32
            raw=key['product'].astype(np.uint64)-ghat*key['rb']-bhat*key['rv']
            if party==0:raw+=ghat*bhat
            raw&=M32
        else:
            local_g=(values[:,0]+key['rg'])&M16
            peer=ch.swap(local_g.astype('<u2').tobytes())
            ghat=(local_g+np.frombuffer(peer,dtype='<u2'))&M16
            u=(ghat+np.uint64(1<<15))&M16;v=(bhat+np.uint64(1<<15))&M16
            c=dcf_eval(key['dcfx'],u,party,16,2,16)
            d=dcf_eval(key['dcfy'],v,party,16,2,16)
            us=(u.astype(np.int64)-(1<<15)).view(np.uint64)
            vs=(v.astype(np.int64)-(1<<15)).view(np.uint64)
            raw=key['product'].astype(np.uint64)-us*key['rb']-vs*key['rg']
            raw+=np.uint64(1<<16)*(c[:,0]*vs+c[:,1]+d[:,0]*us+d[:,1])
            if party==0:raw+=us*vs
            raw&=M32
        after_gate=time.perf_counter_ns();gate_bytes=ch.sent-prefix_bytes;gate_exchanges=ch.exchanges-prefix_exchanges
        if postprocess=='faithful':
            masked=(raw+key['ars_mask'])&M32
            peer=ch.swap(masked.astype('<u4').tobytes())
            opened=(masked+np.frombuffer(peer,dtype='<u4'))&M32
            low=dcf_eval(key['ars_low'],opened&np.uint64(255),party,8,1,64)[:,0]
            high=dcf_eval(key['ars_high'],opened,party,32,1,64)[:,0]
            result=key['ars_const']+low+high
            if party==0:result+=opened>>np.uint64(8)
        else:
            rounded=(raw+key['rounding_zero'])&M32
            if party==0:rounded=(rounded+np.uint64(255))&M32
            rounded>>=np.uint64(8)
            masked=(rounded+key['lift_mask'])&M24
            peer=ch.swap(pack24(masked))
            opened=(masked+unpack24(peer))&M24
            shifted=(opened+np.uint64(1<<23))&M24
            correction=dcf_eval(key['lift_dcf'],shifted,party,24,1,64)[:,0]
            result=key['lift_const']+correction
            if party==0:result+=shifted
        end=time.perf_counter_ns()
        pipe.send({'party':party,'gate_ms':(after_gate-after_prefix)/1e6,'total_ms':(end-start)/1e6,
                   'input_quantization_ms':prefix_ms,'input_quantization_sent_bytes':prefix_bytes,
                   'input_quantization_exchanges':prefix_exchanges,
                   'gate_sent_bytes':gate_bytes,'total_sent_bytes':ch.sent,'gate_exchanges':gate_exchanges,
                   'total_exchanges':ch.exchanges,'quantized_input_shares':quantized,'raw_share':raw,'out_share':result})
    except BaseException as err:pipe.send({'party':party,'error':repr(err)})
    finally:sock.close();pipe.close()


def run(a,b,mode,table_values=None,rtt_ms=0,postprocess='faithful',b_value_bits=16,input_fixed32=False,quantizer_output_bits=32):
    assert mode in ('beaver32','mixed16','fused32')
    if input_fixed32:
        from gated_input_quantizer import plain_quantize,prepare as prepare_quantizer
        original_a=np.ascontiguousarray(a,dtype=np.uint64);original_b=np.ascontiguousarray(b,dtype=np.uint64)
        a=(plain_quantize(original_a,127)&255).astype(np.uint8)
        b=plain_quantize(original_b,32767).view(np.uint64)
    a=np.ascontiguousarray(a,dtype=np.uint8);b=np.ascontiguousarray(b,dtype=np.uint64)
    assert b_value_bits in (16,32) and (mode!='mixed16' or b_value_bits==16)
    value_mask=np.uint64((1<<b_value_bits)-1);sign=np.uint64(1<<(b_value_bits-1))
    signed_b=((b&value_mask)^sign).astype(np.int64)-int(sign)
    b=signed_b.view(np.uint64);n=len(a)
    table=public_table() if table_values is None else np.ascontiguousarray(table_values,dtype=np.uint64)
    assert table.shape in ((256,),(n,256))
    signed_table=table.view(np.int64)
    assert np.all(signed_table>=-32768) and np.all(signed_table<=32767)
    start=time.perf_counter_ns()
    assert postprocess in ('faithful','legacy_stochastic')
    if mode=='mixed16':inputs,keys=prepare_mixed(a,b,rounding=postprocess!='faithful')
    else:
        inputs,keys,_=prepare(a,b,'fused' if mode=='fused32' else 'baseline','native',table,32,rounding=postprocess!='faithful')
    gate_material=material_bytes(keys);lift=prepare_faithful(n) if postprocess=='faithful' else prepare_lift(n)
    for p in (0,1):keys[p].update(lift[p])
    prefix_material=0
    if input_fixed32:
        inputs=list(zip(split64(original_a,32),split64(original_b,32)))
        prefix=prepare_quantizer(2*n,np.concatenate([np.full(n,127),np.full(n,32767)]),output_bits=quantizer_output_bits)
        prefix_material=material_bytes(prefix)
        for p in (0,1):keys[p].update(prefix[p])
    all_material=material_bytes(keys);dealer_ms=(time.perf_counter_ns()-start)/1e6
    ctx=mp.get_context('fork');sockets=socket.socketpair();barrier=ctx.Barrier(2)
    parents=[];procs=[]
    for p in (0,1):
        parent,child=ctx.Pipe(duplex=False)
        proc=ctx.Process(target=worker,args=(p,sockets[p],inputs[p],keys[p],mode,table,rtt_ms,postprocess,barrier,child))
        proc.start();child.close();parents.append(parent);procs.append(proc)
    for s in sockets:s.close()
    results=[]
    for pipe in parents:
        if not pipe.poll(30):
            for proc in procs:proc.terminate()
            raise RuntimeError('worker timeout')
        results.append(pipe.recv())
    for proc in procs:proc.join(30);assert proc.exitcode==0
    if any('error' in r for r in results):raise RuntimeError(results)
    quantized_shares=[r.pop('quantized_input_shares') for r in results]
    if input_fixed32:
        quantizer_mask=np.uint64((1<<quantizer_output_bits)-1)
        qa,qb=np.split((quantized_shares[0]+quantized_shares[1])&quantizer_mask,2)
        assert np.array_equal(qa,plain_quantize(original_a,127).view(np.uint64)&quantizer_mask)
        assert np.array_equal(qb,plain_quantize(original_b,32767).view(np.uint64)&quantizer_mask)
    actual_raw=(results[0].pop('raw_share')+results[1].pop('raw_share'))&M32
    g=signed_table[a] if table.ndim==1 else signed_table[np.arange(n),a]
    expected_product=g*signed_b
    expected_raw=expected_product.view(np.uint64)&M32
    expected_signed=(expected_raw^np.uint64(1<<31)).astype(np.int64)-(1<<31)
    assert np.array_equal(actual_raw,expected_raw)
    out=(results[0].pop('out_share')+results[1].pop('out_share')).view(np.int64)
    floor=expected_signed//256
    if postprocess=='faithful':assert np.array_equal(out,floor)
    else:assert np.all((out==floor)|(out==floor+1))
    total=sum(r['total_sent_bytes'] for r in results)
    post_bytes=8 if postprocess=='faithful' else 6
    assert total==n*((18 if mode=='beaver32' else 10)+post_bytes+(32 if input_fixed32 else 0))
    return {'mode':mode,'n':n,'b_value_bits':b_value_bits,'raw_correct':True,'signed64_output_correct':True,'postprocess':postprocess,
            'input_fixed32':input_fixed32,'input_quantizer_key_array_bytes_both_parties':prefix_material,
            'quantizer_output_bits':quantizer_output_bits if input_fixed32 else None,
            'input_quantizer_correct':True if input_fixed32 else None,
            'emulated_rtt_ms_per_logical_exchange':rtt_ms,'parties':results,
            'online_sent_bytes_both_parties':total,'dealer_ms':dealer_ms,
            'gate_key_array_bytes_both_parties':gate_material,'total_key_array_bytes_both_parties':all_material,
            'public_table_bytes':table.nbytes}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rtt-ms',default='0,50')
    parser.add_argument('--sizes',default='128,4864')
    parser.add_argument('--postprocess',choices=('faithful','legacy_stochastic'),default='faithful');args=parser.parse_args()
    rng=np.random.default_rng(291801);checks=[]
    b=np.tile(np.array([-32768,-32767,-1,0,1,32766,32767],dtype=np.int64),256).view(np.uint64)
    a=np.repeat(np.arange(256,dtype=np.uint8),7)
    table=np.resize(np.array([-32768,-32767,-1,0,1,32766,32767],dtype=np.int64),256).view(np.uint64)
    for mode in ('beaver32','mixed16','fused32'):checks.append(run(a,b,mode,table,postprocess=args.postprocess))
    rows=[]
    for n in map(int,args.sizes.split(',')):
        for rtt in map(float,args.rtt_ms.split(',')):
            for repetition in range(3):
                a=rng.integers(0,256,n,dtype=np.uint8)
                b=rng.integers(-32768,32768,n,dtype=np.int64).view(np.uint64)
                orders=(('beaver32','mixed16','fused32'),('mixed16','fused32','beaver32'),('fused32','beaver32','mixed16'))
                for mode in orders[repetition]:
                    record=run(a,b,mode,rtt_ms=rtt,postprocess=args.postprocess);record.update(repetition=repetition,order=list(orders[repetition]));rows.append(record)
            print('finished',n,'emulated RTT',rtt,flush=True)
    out={'description':'Common start is a8 shares and ring32 b shares with signed16-value promise; all end with signed64 shares of rounded finite LUT product',
         'postprocess':args.postprocess,
         'boundary_checks':checks,'measurements':rows,'native_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
         'prior':'LLAMA signed multiplication and signed extension are existing constructions; DCF correction-value algebra adapted under upstream MIT notice.',
         'limits':'Trusted parent/fork harness owns all secrets. Emulated delay on local UNIX sockets, not actual WAN. No input quantization, model projections or dealer network. Online output reconstruction to the harness excluded. Key sizes count actual array storage, including32-bit storage for24-bit masks.'}
    filename='gated_lut_strong_baselines_faithful.json' if args.postprocess=='faithful' else 'gated_lut_strong_baselines_legacy_stochastic.json'
    (ROOT/'results'/filename).write_text(json.dumps(out,indent=2)+'\n')
    for n in map(int,args.sizes.split(',')):
        for rtt in map(float,args.rtt_ms.split(',')):
            for mode in ('beaver32','mixed16','fused32'):
                r=[v for v in rows if v['n']==n and v['emulated_rtt_ms_per_logical_exchange']==rtt and v['mode']==mode]
                print(n,rtt,mode,'ms',np.median([v['parties'][0]['total_ms'] for v in r]),'bytes',r[0]['online_sent_bytes_both_parties'],'keys',r[0]['total_key_array_bytes_both_parties'])


if __name__=='__main__':main()
