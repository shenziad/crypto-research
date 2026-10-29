"""Local finite gate->private contraction->faithful floor protocol.

Fresh dealer material. Normalized fixed8 ring32 input shares; private signed16
down weights; exact signed64 late-floor output. All rings between quantizer and
final shift are ring56, with actual seven-byte wire encoding. Standard FSS,
LLAMA mixed multiplication and Beaver matrix triples, not new primitives.
"""
import json
import multiprocessing as mp
import os
import socket
import time
import numpy as np
from fused_gated_lut import ROOT,random64,split64,split8
from gated_input_quantizer import prepare as prepare_quantizer,prepare_pair,online as quantize_online,plain_quantize
from gated_lut_strong_baselines import Channel,material_bytes,prepare_mixed
from fused_gated_lut import prepare as prepare_base
from dpf_aes_native import generate_batch as dpf_gen,lookup_batch
from dcf_aes_native import generate_batch as dcf_gen,evaluate_batch as dcf_eval

MASK=np.uint64((1<<56)-1);M16=np.uint64(65535)


def pack56(x):
    return np.ascontiguousarray(x,dtype='<u8').view(np.uint8).reshape(-1,8)[:,:7].copy().tobytes()


def unpack56(packet):
    src=np.frombuffer(packet,dtype=np.uint8).reshape(-1,7)
    out=np.zeros((len(src),8),dtype=np.uint8);out[:,:7]=src
    return out.view('<u8').reshape(-1)


def split56(x):return [v&MASK for v in split64(x)]


def prepare_shift(n,shift):
    assert 1<=shift<56
    r=random64((n,))&MASK;rs=split56(r)
    const=-(r>>np.uint64(shift))-np.where(r<np.uint64(1<<55),np.uint64(1<<(56-shift)),np.uint64(0))
    cs=split64(const)
    low=dcf_gen(r&np.uint64((1<<shift)-1),np.full((n,1),(1<<64)-1,dtype=np.uint64),shift,64)
    high=dcf_gen((r+np.uint64(1<<55))&MASK,np.full((n,1),1<<(56-shift),dtype=np.uint64),56,64)
    return [dict(s_mask=rs[p],s_const=cs[p],s_low=low[p],s_high=high[p]) for p in (0,1)]


def prepare_gate(n,mode):
    if mode in ('beaver32_lift','mixed16x16_lift'):
        zero=np.zeros(n,dtype=np.uint64);azero=np.zeros(n,dtype=np.uint8)
        if mode=='mixed16x16_lift':_,original=prepare_mixed(azero,zero,rounding=False)
        else:_,original,_=prepare_base(azero,zero,'baseline','native',ring_bits=32,rounding=False)
        keys=[]
        for p in (0,1):keys.append({'g_'+('rg' if k=='rv' else k):v for k,v in original[p].items()})
        r=random64((n,))&np.uint64((1<<32)-1);rs=split64(r,32)
        const=split64(-r-np.where(r<np.uint64(1<<31),np.uint64(1<<32),np.uint64(0)))
        dcf=dcf_gen((r+np.uint64(1<<31))&np.uint64((1<<32)-1),np.full((n,1),1<<32,dtype=np.uint64),32,64)
        for p in (0,1):keys[p].update(l_mask=rs[p],l_const=const[p],l_dcf=dcf[p])
        return keys
    ra=np.frombuffer(os.urandom(n),dtype=np.uint8).copy();ras=split8(ra)
    rb=random64((n,))&MASK;rbs=split56(rb)
    if mode=='fused56':
        dpf=dpf_gen(ra,np.stack([np.ones(n,dtype=np.uint64),-rb],axis=1),group_bits=64)
        return [dict(g_ra=ras[p],g_rb=rbs[p],g_dpf=dpf[p]) for p in (0,1)]
    rg=random64((n,))&(M16 if mode=='mixed16x56' else MASK)
    rgs=split56(rg);products=split56(rg*rb)
    dpf=dpf_gen(ra,np.ones((n,1),dtype=np.uint64),group_bits=16 if mode=='mixed16x56' else 64)
    keys=[dict(g_ra=ras[p],g_rb=rbs[p],g_rg=rgs[p],g_product=products[p],g_dpf=dpf[p]) for p in (0,1)]
    if mode=='mixed16x56':
        dcfs=dcf_gen(rg,np.stack([np.ones(n,dtype=np.uint64),-rb],axis=1),16,64)
        for p in (0,1):keys[p]['g_dcf']=dcfs[p]
    return keys


def worker(party,sock,inputs,key,mode,table,shape,shift,rtt,barrier,pipe):
    try:
        af,bf,wshare=inputs;n=len(af);m=shape[0];ch=Channel(sock,party,rtt)
        # Model-mask initialization is measured separately, never silently free.
        t0=time.perf_counter_ns();masked_w=(wshare+key['mat_a'])&MASK
        peer=ch.swap(pack56(masked_w));public_w=(masked_w+unpack56(peer).reshape(shape))&MASK
        init_ms=(time.perf_counter_ns()-t0)/1e6;init_bytes=ch.sent
        ch.sent=0;ch.exchanges=0;barrier.wait();start=time.perf_counter_ns()
        quantized=quantize_online(party,ch,np.concatenate([af,bf]),key)
        a=(quantized[:n]&np.uint64(255)).astype(np.uint8);b=quantized[n:]&MASK
        narrow=mode in ('beaver32_lift','mixed16x16_lift')
        mixed=mode in ('mixed16x56','mixed16x16_lift')
        gate_mask=M16 if mode=='mixed16x16_lift' else np.uint64((1<<32)-1) if narrow else MASK
        t1=time.perf_counter_ns()
        la=np.add(a,key['g_ra'],dtype=np.uint8);lb=(b+key['g_rb'])&gate_mask
        pack_gate=lambda x:x.astype('<u2' if mode=='mixed16x16_lift' else '<u4').tobytes() if narrow else pack56(x)
        unpack_gate=lambda x:np.frombuffer(x,dtype='<u2' if mode=='mixed16x16_lift' else '<u4') if narrow else unpack56(x)
        peer=ch.swap(la.tobytes()+pack_gate(lb))
        ahat=np.add(la,np.frombuffer(peer,dtype=np.uint8,count=n),dtype=np.uint8)
        bhat=(lb+unpack_gate(peer[n:]))&gate_mask
        values=lookup_batch(key['g_dpf'],ahat,table,party,2 if mode=='fused56' else 1,
            group_bits=16 if mixed else 32 if narrow else 64)
        if mode=='fused56':raw=(bhat*values[:,0]+values[:,1])&MASK
        else:
            gm=(values[:,0]+key['g_rg'])&(M16 if mixed else gate_mask)
            peer=ch.swap(gm.astype('<u2').tobytes() if mixed else pack_gate(gm))
            gh=(gm+(np.frombuffer(peer,dtype='<u2') if mixed else unpack_gate(peer)))&(M16 if mixed else gate_mask)
            if mode=='mixed16x16_lift':
                u=(gh+np.uint64(32768))&M16;v=(bhat+np.uint64(32768))&M16
                cx=dcf_eval(key['g_dcfx'],u,party,16,2,16);cy=dcf_eval(key['g_dcfy'],v,party,16,2,16)
                us=(u.astype(np.int64)-32768).view(np.uint64);vs=(v.astype(np.int64)-32768).view(np.uint64)
                raw=key['g_product'].astype(np.uint64)-us*key['g_rb']-vs*key['g_rg']
                raw+=np.uint64(65536)*(cx[:,0]*vs+cx[:,1]+cy[:,0]*us+cy[:,1])
                if party==0:raw+=us*vs
            elif mode=='mixed16x56':
                u=(gh+np.uint64(32768))&M16;us=(u.astype(np.int64)-32768).view(np.uint64)
                correction=dcf_eval(key['g_dcf'],u,party,16,2,64)
                raw=key['g_product']-us*key['g_rb']-bhat*key['g_rg']
                raw+=np.uint64(65536)*(correction[:,0]*bhat+correction[:,1])
                if party==0:raw+=us*bhat
            else:
                raw=key['g_product'].astype(np.uint64)-gh*key['g_rb']-bhat*key['g_rg']
                if party==0:raw+=gh*bhat
            raw&=MASK
        if narrow:
            mask32=np.uint64((1<<32)-1);masked=(raw+key['l_mask'])&mask32
            peer=ch.swap(masked.astype('<u4').tobytes());h=(masked+np.frombuffer(peer,dtype='<u4'))&mask32
            raw=key['l_const']+dcf_eval(key['l_dcf'],h,party,32,1,64)[:,0]
            if party==0:raw+=h
            raw&=MASK
        t2=time.perf_counter_ns()
        masked=(raw+key['mat_v'])&MASK;peer=ch.swap(pack56(masked))
        opened=(masked+unpack56(peer))&MASK
        acc=key['mat_product']-public_w@key['mat_v']-key['mat_a']@opened
        if party==0:acc+=public_w@opened
        acc&=MASK;t3=time.perf_counter_ns()
        masked=(acc+key['s_mask'])&MASK;peer=ch.swap(pack56(masked))
        opened=(masked+unpack56(peer))&MASK
        low=dcf_eval(key['s_low'],opened&np.uint64((1<<shift)-1),party,shift,1,64)[:,0]
        high=dcf_eval(key['s_high'],opened,party,56,1,64)[:,0]
        out=key['s_const']+low+high
        if party==0:out+=opened>>np.uint64(shift)
        end=time.perf_counter_ns()
        pipe.send(dict(party=party,total_ms=(end-start)/1e6,quantizer_ms=(t1-start)/1e6,
            gate_ms=(t2-t1)/1e6,projection_ms=(t3-t2)/1e6,final_shift_ms=(end-t3)/1e6,
            initialization_ms=init_ms,initialization_sent_bytes=init_bytes,online_sent_bytes=ch.sent,
            online_exchanges=ch.exchanges,quantized=quantized,raw=raw,acc=acc,out=out))
    except BaseException as err:pipe.send(dict(party=party,error=repr(err)))
    finally:sock.close();pipe.close()


def run(af,bf,w,table,mode,rtt=0,shift=20):
    assert mode in ('beaver56','mixed16x56','fused56','beaver32_lift','mixed16x16_lift')
    af=np.ascontiguousarray(af,dtype=np.uint64);bf=np.ascontiguousarray(bf,dtype=np.uint64)
    w=np.ascontiguousarray(w,dtype=np.int64);n=len(af);m=w.shape[0]
    assert w.shape==(m,n) and np.all(w>=-32768) and np.all(w<=32767)
    table=np.ascontiguousarray(table,dtype=np.int64);assert table.shape==(256,) and np.all(table>=-32768) and np.all(table<=32767)
    bound=int(np.abs(w).sum(axis=1).max())*max(abs(int(v)) for v in table)*32767
    assert bound<(1<<55)
    start=time.perf_counter_ns()
    narrow=mode in ('beaver32_lift','mixed16x16_lift')
    keys=prepare_quantizer(2*n,np.concatenate([np.full(n,127),np.full(n,32767)])) if narrow else prepare_pair(n)
    prefix_bytes=material_bytes(keys);gate=prepare_gate(n,mode);gate_bytes=material_bytes(gate)
    shifts=prepare_shift(m,shift);shift_bytes=material_bytes(shifts)
    a=random64(w.shape)&MASK;v=random64((n,))&MASK;product=(a@v)&MASK
    ma=split56(a);mv=split56(v);mt=split56(product)
    for p in (0,1):keys[p].update(gate[p]);keys[p].update(shifts[p]);keys[p].update(mat_a=ma[p],mat_v=mv[p],mat_product=mt[p])
    material=material_bytes(keys);dealer_ms=(time.perf_counter_ns()-start)/1e6
    wa=split56(w.view(np.uint64));aa=split64(af,32);ba=split64(bf,32)
    ctx=mp.get_context('fork');sockets=socket.socketpair();barrier=ctx.Barrier(2);parents=[];procs=[]
    for p in (0,1):
        parent,child=ctx.Pipe(duplex=False)
        proc=ctx.Process(target=worker,args=(p,sockets[p],(aa[p],ba[p],wa[p]),keys[p],mode,table.view(np.uint64),w.shape,shift,rtt,barrier,child))
        proc.start();child.close();parents.append(parent);procs.append(proc)
    for s in sockets:s.close()
    rows=[]
    for pipe in parents:
        if not pipe.poll(60):
            for proc in procs:proc.terminate()
            raise RuntimeError('block worker timeout')
        rows.append(pipe.recv())
    for proc in procs:proc.join(30);assert proc.exitcode==0
    if any('error' in r for r in rows):raise RuntimeError(rows)
    qa=plain_quantize(af,127);qb=plain_quantize(bf,32767)
    q_actual=rows[0].pop('quantized')+rows[1].pop('quantized')
    assert np.array_equal(q_actual[:n]&np.uint64((1<<32)-1),qa.view(np.uint64)&np.uint64((1<<32)-1))
    qmask=np.uint64((1<<32)-1) if narrow else np.uint64((1<<64)-1)
    assert np.array_equal(q_actual[n:]&qmask,qb.view(np.uint64)&qmask)
    raw=table[qa&255]*qb
    assert np.array_equal((rows[0].pop('raw')+rows[1].pop('raw'))&MASK,raw.view(np.uint64)&MASK)
    acc=w@raw
    assert np.array_equal((rows[0].pop('acc')+rows[1].pop('acc'))&MASK,acc.view(np.uint64)&MASK)
    assert np.array_equal((rows[0].pop('out')+rows[1].pop('out')).view(np.int64),acc//(1<<shift))
    gate_online={'beaver56':30,'mixed16x56':20,'fused56':16,'beaver32_lift':26,'mixed16x16_lift':18}[mode]
    online=sum(r['online_sent_bytes'] for r in rows)
    assert online==n*(32+gate_online+14)+m*14
    assert sum(r['initialization_sent_bytes'] for r in rows)==m*n*14
    return dict(mode=mode,n=n,m=m,shift=shift,emulated_rtt_ms=rtt,all_functional_checks_passed=True,
        signed56_all_domain_accumulator_bound=bound,parties=rows,online_sent_bytes_both_parties=online,
        model_initialization_sent_bytes_both_parties=m*n*14,dealer_ms=dealer_ms,
        key_array_bytes_both_parties=material,input_quantizer_key_bytes=prefix_bytes,gate_key_bytes=gate_bytes,
        final_shift_key_bytes=shift_bytes,matrix_triple_key_bytes=material-prefix_bytes-gate_bytes-shift_bytes)


if __name__=='__main__':
    rng=np.random.default_rng(2026102914);n,m=64,16
    af=rng.integers(0,1<<32,n,dtype=np.uint64);bf=rng.integers(0,1<<32,n,dtype=np.uint64)
    table=np.resize(np.array([-32768,-32767,-1,0,1,32766,32767],dtype=np.int64),256)
    w=rng.integers(-32767,32768,(m,n),dtype=np.int64)
    records=[run(af,bf,w,table,mode) for mode in ('beaver56','mixed16x56','beaver32_lift','mixed16x16_lift','fused56')]
    (ROOT/'results/gated_contraction_synthetic.json').write_text(json.dumps(dict(results=records,
        limits='Local trusted-parent test. Wide quantizer used on both branches. Model-mask setup counted separately; no claim of independently novel primitives or secure full LLM.'),indent=2)+'\n')
    for row in records:print(row['mode'],row['all_functional_checks_passed'],row['online_sent_bytes_both_parties'],row['key_array_bytes_both_parties'])
