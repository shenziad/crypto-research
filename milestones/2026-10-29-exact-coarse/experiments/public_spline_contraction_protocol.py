"""Local matched public-basis/long-payload spline through private contraction.

Same finite function: normalized A in ring63 (signed32 value promise by default),
normalized fixed8 B in ring32, private signed16 down matrix -> signed64 output.
Public splineF22, B half-up clip16, private integer dot, exact finalshift34.
"""
import multiprocessing as mp
import socket
import time
import numpy as np
from fused_gated_lut import ROOT,random64,split64
from gated_lut_strong_baselines import Channel,material_bytes
from gated_input_quantizer import prepare_round,plain_quantize
from dcf_aes_native import generate_batch,evaluate_batch
from public_spline_fss import prepare as clip_prepare,evaluate as clip_evaluate
from public_spline_coefficient_basis import prepare as spline_prepare,evaluate as spline_evaluate
from public_silu_spline import evaluate as plain_spline
from public_spline_selector import prepare as selector_prepare,evaluate as selector_evaluate

MASK=np.uint64((1<<63)-1);M32=np.uint64((1<<32)-1)


class PayloadDelayChannel(Channel):
    """Explicit full-duplex per-message payload-delay model, not a WAN emulator."""
    bandwidth_mbps=0
    def swap(self,packet):
        other=super().swap(packet)
        if self.bandwidth_mbps:time.sleep(8*len(packet)/(self.bandwidth_mbps*1e6))
        return other


def split63(v):return [a&MASK for a in split64(v)]


def clip_spec(maximum=32767):
    ring=1<<32
    return dict(input_bits=32,function_at_last=-1,delta_runs=[(1,maximum,1),(ring//2,ring//2,-2*maximum),(ring-maximum+1,ring-1,1)])


def shift_prepare(n,shift):
    r=random64((n,))&MASK;rs=split63(r)
    cs=split64(-(r>>np.uint64(shift))-np.where(r<np.uint64(1<<62),np.uint64(1<<(63-shift)),np.uint64(0)))
    low=generate_batch(r&np.uint64((1<<shift)-1),np.full((n,1),(1<<64)-1,dtype=np.uint64),shift,64)
    high=generate_batch((r+np.uint64(1<<62))&MASK,np.full((n,1),1<<(63-shift),dtype=np.uint64),63,64)
    return [dict(mask=rs[p],constant=cs[p],low=low[p],high=high[p]) for p in (0,1)]


def worker(p,sock,inputs,key,variant,spec,shape,shift,rtt,bandwidth_mbps,barrier,pipe):
    try:
        a,b,w=inputs;n=len(a);ch=PayloadDelayChannel(sock,p,rtt)
        sk={k[2:]:v for k,v in key.items() if k.startswith('s_')}
        ck={k[2:]:v for k,v in key.items() if k.startswith('c_')}
        qk={k[2:]:v for k,v in key.items() if k.startswith('q_')}
        start=time.perf_counter_ns();masked_w=(w+key['mat_a'])&MASK
        peer=ch.swap(masked_w.astype('<u8').tobytes());public_w=(masked_w+np.frombuffer(peer,dtype='<u8').reshape(shape))&MASK
        init_ms=(time.perf_counter_ns()-start)/1e6;init_bytes=ch.sent;ch.sent=0;ch.exchanges=0
        ch.bandwidth_mbps=bandwidth_mbps
        barrier.wait();start=time.perf_counter_ns()
        am=(a+sk['input_mask'])&MASK;bm=(b+qk['q_mask'])&M32
        peer=ch.swap(am.astype('<u8').tobytes()+bm.astype('<u4').tobytes())
        ah=(am+np.frombuffer(peer,dtype='<u8',count=n))&MASK
        bh=(bm+np.frombuffer(peer,dtype='<u4',count=n,offset=8*n))&M32
        low=evaluate_batch(qk['q_low'],bh&np.uint64(255),p,8,1,32)[:,0]
        high=evaluate_batch(qk['q_high'],bh,p,32,1,32)[:,0]
        rounded=qk['q_const'].astype(np.uint64)+low+high
        if p==0:rounded+=bh>>np.uint64(8)
        rounded&=M32
        coefficients=None;gvec=None
        if spec.get('use_dither_fss',False):
            from dithered_spline_fss import evaluate as dither_evaluate
            values=dither_evaluate(sk,ah,spec,p,63,variant)
            if variant=='selector':coefficients=values
            elif variant=='weighted':gvec=values;g=gvec[:,0]
            else:g=values
        elif variant=='selector':coefficients=selector_evaluate(sk,ah,spec,p,63,False)
        elif variant=='weighted':gvec=selector_evaluate(sk,ah,spec,p,63,True);g=gvec[:,0]
        else:g=spline_evaluate(sk,ah,spec,p,63,cached=3,variant=variant)
        t1=time.perf_counter_ns()
        cm=(rounded+ck['input_mask'])&M32
        if variant=='weighted':packet=cm.astype('<u4').tobytes()
        else:
            gm=((coefficients[:,0]+sk['horner_rs']) if variant=='selector' else (g+key['mul_rg']))&MASK
            packet=cm.astype('<u4').tobytes()+gm.astype('<u8').tobytes()
        peer=ch.swap(packet)
        chb=(cm+np.frombuffer(peer,dtype='<u4',count=n))&M32
        if variant!='weighted':gh=(gm+np.frombuffer(peer,dtype='<u8',count=n,offset=4*n))&MASK
        bq=clip_evaluate(ck,chb,clip_spec(),p,bits=32,cached=2)&MASK
        if variant=='selector':
            g=sk['horner_product']-ah*sk['horner_rs']-gh*sk['input_mask']+coefficients[:,1]+sk['refresh']
            if p==0:g+=ah*gh
            g&=MASK
        t2=time.perf_counter_ns()
        bm=(bq+(sk['b_mask'] if variant=='weighted' else key['mul_rb']))&MASK
        packet=bm.astype('<u8').tobytes()
        if variant=='selector':
            gm=(g+key['mul_rg'])&MASK;packet+=gm.astype('<u8').tobytes()
        peer=ch.swap(packet);bh=(bm+np.frombuffer(peer,dtype='<u8',count=n))&MASK
        if variant=='selector':gh=(gm+np.frombuffer(peer,dtype='<u8',count=n,offset=8*n))&MASK
        if variant=='weighted':raw=bh*gvec[:,0]+gvec[:,1]
        else:
            raw=key['mul_product']-gh*key['mul_rb']-bh*key['mul_rg']
            if p==0:raw+=gh*bh
        raw&=MASK;t3=time.perf_counter_ns()
        zm=(raw+key['mat_v'])&MASK;peer=ch.swap(zm.astype('<u8').tobytes());zh=(zm+np.frombuffer(peer,dtype='<u8'))&MASK
        acc=key['mat_product']-public_w@key['mat_v']-key['mat_a']@zh
        if p==0:acc+=public_w@zh
        acc&=MASK;t4=time.perf_counter_ns()
        sm=(acc+key['out_mask'])&MASK;peer=ch.swap(sm.astype('<u8').tobytes());sh=(sm+np.frombuffer(peer,dtype='<u8'))&MASK
        lo=evaluate_batch(key['out_low'],sh&np.uint64((1<<shift)-1),p,shift,1,64)[:,0]
        hi=evaluate_batch(key['out_high'],sh,p,63,1,64)[:,0]
        out=key['out_constant']+lo+hi
        if p==0:out+=sh>>np.uint64(shift)
        end=time.perf_counter_ns()
        pipe.send(dict(party=p,total_ms=(end-start)/1e6,first_round_spline_ms=(t1-start)/1e6,
            clip_ms=(t2-t1)/1e6,multiply_ms=(t3-t2)/1e6,projection_ms=(t4-t3)/1e6,final_shift_ms=(end-t4)/1e6,
            initialization_ms=init_ms,initialization_sent_bytes=init_bytes,online_sent_bytes=ch.sent,online_exchanges=ch.exchanges,
            rounded=rounded,clipped=bq,g=g,weighted_aux=gvec[:,1] if variant=='weighted' else None,raw=raw,acc=acc,out=out))
    except BaseException as err:pipe.send(dict(party=p,error=repr(err)))
    finally:sock.close();pipe.close()


def run(a,b,w,spec,variant,rtt=0,gpu_output=None,bandwidth_mbps=0,start_method='fork',backend='cpu'):
    assert variant in ('basis','wide','selector','weighted')
    assert bandwidth_mbps>=0 and rtt>=0
    assert backend in ('cpu','cuda')
    target=worker
    if backend=='cuda':
        assert start_method=='spawn','CUDA workers require independent process startup'
        from public_spline_cuda_protocol import worker_cuda
        target=worker_cuda
    a=np.ascontiguousarray(a,dtype=np.int64);b=np.ascontiguousarray(b,dtype=np.uint64);w=np.ascontiguousarray(w,dtype=np.int64)
    n=len(a);m=w.shape[0];view=spec.get('input_bits',63);shift=spec['fraction_bits']+12
    assert w.shape==(m,n) and np.all(w>=-32768) and np.all(w<=32767)
    promise=spec.get('promised_input_bits',view) if spec.get('use_dither_fss',False) else view
    assert np.all(a>=-(1<<(promise-1))) and np.all(a<(1<<(promise-1)))
    bound=int(np.abs(w).sum(axis=1).max())*spec['absolute_bound']*32768;assert bound<(1<<62)
    t0=time.perf_counter_ns()
    if spec.get('use_dither_fss',False):
        from dithered_spline_fss import prepare as dither_prepare
        sk,_=dither_prepare(n,spec,63,variant=variant)
    elif variant in ('selector','weighted'):sk,_=selector_prepare(n,spec,63,weighted=variant=='weighted')
    else:sk,_=spline_prepare(n,spec,63,variant=variant)
    spline_bytes=material_bytes(sk)
    qk=prepare_round(n);ck,_=clip_prepare(n,32);ok=shift_prepare(m,shift)
    keys=[{} for _ in (0,1)]
    for p in (0,1):
        for prefix,source in (('s_',sk[p]),('q_',qk[p]),('c_',ck[p]),('out_',ok[p])):
            keys[p].update({prefix+k:v for k,v in source.items()})
    if variant!='weighted':
        rg=random64((n,))&MASK;rb=random64((n,))&MASK
        rgs=split63(rg);rbs=split63(rb);products=split63(rg*rb)
    A=random64(w.shape)&MASK;v=random64((n,))&MASK
    As=split63(A);vs=split63(v);ts=split63((A@v)&MASK)
    for p in (0,1):
        keys[p].update(mat_a=As[p],mat_v=vs[p],mat_product=ts[p])
        if variant!='weighted':keys[p].update(mul_rg=rgs[p],mul_rb=rbs[p],mul_product=products[p])
    material=material_bytes(keys);dealer_ms=(time.perf_counter_ns()-t0)/1e6
    aa=split63(a.view(np.uint64));bb=split64(b,32);ww=split63(w.view(np.uint64))
    assert start_method in ('fork','spawn')
    ctx=mp.get_context(start_method);sockets=socket.socketpair();barrier=ctx.Barrier(2);parents=[];procs=[]
    for p in (0,1):
        parent,child=ctx.Pipe(duplex=False);proc=ctx.Process(target=target,args=(p,sockets[p],(aa[p],bb[p],ww[p]),keys[p],variant,spec,w.shape,shift,rtt,bandwidth_mbps,barrier,child))
        proc.start();child.close();parents.append(parent);procs.append(proc)
    for sock in sockets:sock.close()
    rows=[]
    for pipe in parents:
        if not pipe.poll(60):
            for proc in procs:proc.terminate()
            raise RuntimeError('spline block timeout')
        rows.append(pipe.recv())
    for proc in procs:proc.join(30);assert proc.exitcode==0
    if any('error' in r for r in rows):raise RuntimeError(rows)
    def reconstruct(field):return rows[0].pop(field)+rows[1].pop(field)
    bs=((b&M32)^np.uint64(1<<31)).astype(np.int64)-(1<<31)
    assert np.array_equal(reconstruct('rounded')&M32,((bs+128)//256).view(np.uint64)&M32)
    bi=plain_quantize(b,32767)
    assert np.array_equal(reconstruct('clipped')&MASK,bi.view(np.uint64)&MASK)
    if spec.get('use_dither_fss',False):
        from dithered_silu import evaluate as dither_plain
        parent_h=(a.view(np.uint64)+sk[0]['input_mask']+sk[1]['input_mask'])&MASK
        rho=(parent_h&np.uint64((1<<spec['classification_shift'])-1)).astype(np.int64)
        gi=dither_plain(spec,a,rho)
    else:gi=plain_spline(spec,a)
    assert np.array_equal(reconstruct('g')&MASK,gi.view(np.uint64)&MASK)
    if variant=='weighted':
        rb=(sk[0]['b_mask']+sk[1]['b_mask'])&MASK
        assert np.array_equal(reconstruct('weighted_aux')&MASK,(-rb*gi.view(np.uint64))&MASK)
    else:
        for r in rows:r.pop('weighted_aux')
    raw=gi*bi;assert np.array_equal(reconstruct('raw')&MASK,raw.view(np.uint64)&MASK)
    acc=w@raw;assert np.array_equal(reconstruct('acc')&MASK,acc.view(np.uint64)&MASK)
    output=reconstruct('out').view(np.int64);assert np.array_equal(output,acc//(1<<shift))
    if gpu_output is not None:assert np.array_equal(output,gpu_output)
    online=sum(r['online_sent_bytes'] for r in rows);assert online==n*(64 if variant=='weighted' else 96 if variant=='selector' else 80)+m*16
    init=sum(r['initialization_sent_bytes'] for r in rows);assert init==n*m*16
    assert all(r['online_exchanges']==5 for r in rows)
    if spec.get('use_dither_fss',False):
        from dithered_spline_fss import classifier_bits
        view=classifier_bits(spec)
    return dict(variant=variant,n=n,m=m,classifier_bits=view,input_value_bits=promise,body_ring_bits=63,emulated_rtt_ms=rtt,
        use_dither_fss=spec.get('use_dither_fss',False),full_classifier_reference=spec.get('full_classifier_reference',False),
        process_context=start_method,backend=backend,worker_arguments='Only own input shares/key dictionary, public specification/shape and IPC handles; spawn avoids inherited parent address space. Trusted test parent retains plaintext reference.',
        payload_delay_mbps_per_direction=bandwidth_mbps,payload_delay_applies_to='online only; model initialization excluded from timing model',
        all_functional_checks_passed=True,gpu_capture_output_checked=gpu_output is not None,parties=rows,
        online_sent_bytes_both_parties=online,model_initialization_sent_bytes_both_parties=init,
        total_key_array_bytes_both_parties=material,spline_key_array_bytes_both_parties=spline_bytes,
        dealer_ms=dealer_ms,all_domain_integer_accumulator_bound=bound)
