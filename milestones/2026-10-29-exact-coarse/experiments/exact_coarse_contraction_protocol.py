"""Exact aligned classifier/body block, five exchanges for basis and weighted.

One low-bit correction is amortized across all interval endpoints. Masked
coarse input is packed with B-clamp opening; gate opening moves to B opening.
"""
import multiprocessing as mp
import socket
import time
import numpy as np
from fused_gated_lut import random64,split64
from gated_lut_strong_baselines import material_bytes
from gated_input_quantizer import prepare_round,plain_quantize
from dcf_aes_native import evaluate_batch
from public_spline_fss import prepare as clip_prepare,evaluate as clip_evaluate
from public_silu_spline import evaluate as plain_spline
from public_spline_contraction_protocol import PayloadDelayChannel,clip_spec,shift_prepare,split63,MASK,M32
from exact_coarse_spline import prepare,masked_classifier,evaluate


def worker(p,sock,inputs,key,variant,spec,shape,shift,rtt,bandwidth,barrier,pipe):
    try:
        a,b,w=inputs;n=len(a);ch=PayloadDelayChannel(sock,p,rtt);vm=np.uint64((1<<(spec['input_bits']-spec['classification_shift']))-1)
        sk={k[2:]:v for k,v in key.items() if k.startswith('s_')};ck={k[2:]:v for k,v in key.items() if k.startswith('c_')};qk={k[2:]:v for k,v in key.items() if k.startswith('q_')}
        start=time.perf_counter_ns();wm=(w+key['mat_a'])&MASK;peer=ch.swap(wm.astype('<u8').tobytes());wh=(wm+np.frombuffer(peer,dtype='<u8').reshape(shape))&MASK
        init_ms=(time.perf_counter_ns()-start)/1e6;init_bytes=ch.sent;ch.sent=0;ch.exchanges=0;ch.bandwidth_mbps=bandwidth
        barrier.wait();start=time.perf_counter_ns()
        am=(a+sk['input_mask'])&MASK;bm=(b+qk['q_mask'])&M32
        peer=ch.swap(am.astype('<u8').tobytes()+bm.astype('<u4').tobytes());ah=(am+np.frombuffer(peer,dtype='<u8',count=n))&MASK;bh=(bm+np.frombuffer(peer,dtype='<u4',count=n,offset=8*n))&M32
        rounded=qk['q_const'].astype(np.uint64)+evaluate_batch(qk['q_low'],bh&np.uint64(255),p,8,1,32)[:,0]+evaluate_batch(qk['q_high'],bh,p,32,1,32)[:,0]
        if p==0:rounded+=bh>>np.uint64(8)
        rounded&=M32;qs=masked_classifier(sk,ah,spec,p);t1=time.perf_counter_ns()
        cm=(rounded+ck['input_mask'])&M32;peer=ch.swap(cm.astype('<u4').tobytes()+qs.astype('<u4').tobytes())
        chb=(cm+np.frombuffer(peer,dtype='<u4',count=n))&M32;th=(qs+np.frombuffer(peer,dtype='<u4',count=n,offset=4*n))&vm
        bq=clip_evaluate(ck,chb,clip_spec(),p,bits=32,cached=2)&MASK
        values=evaluate(sk,ah,th,spec,p,63,variant);gvec=values if variant=='weighted' else None;g=values[:,0] if variant=='weighted' else values;t2=time.perf_counter_ns()
        bm=(bq+(sk['b_mask'] if variant=='weighted' else key['mul_rb']))&MASK;packet=bm.astype('<u8').tobytes()
        if variant=='basis':gm=(g+key['mul_rg'])&MASK;packet+=gm.astype('<u8').tobytes()
        peer=ch.swap(packet);bh=(bm+np.frombuffer(peer,dtype='<u8',count=n))&MASK
        if variant=='weighted':raw=bh*gvec[:,0]+gvec[:,1]
        else:
            gh=(gm+np.frombuffer(peer,dtype='<u8',count=n,offset=8*n))&MASK;raw=key['mul_product']-gh*key['mul_rb']-bh*key['mul_rg']
            if p==0:raw+=gh*bh
        raw&=MASK;t3=time.perf_counter_ns();zm=(raw+key['mat_v'])&MASK;peer=ch.swap(zm.astype('<u8').tobytes());zh=(zm+np.frombuffer(peer,dtype='<u8'))&MASK
        acc=key['mat_product']-wh@key['mat_v']-key['mat_a']@zh
        if p==0:acc+=wh@zh
        acc&=MASK;t4=time.perf_counter_ns();sm=(acc+key['out_mask'])&MASK;peer=ch.swap(sm.astype('<u8').tobytes());sh=(sm+np.frombuffer(peer,dtype='<u8'))&MASK
        out=key['out_constant']+evaluate_batch(key['out_low'],sh&np.uint64((1<<shift)-1),p,shift,1,64)[:,0]+evaluate_batch(key['out_high'],sh,p,63,1,64)[:,0]
        if p==0:out+=sh>>np.uint64(shift)
        end=time.perf_counter_ns()
        pipe.send(dict(party=p,total_ms=(end-start)/1e6,round1_ms=(t1-start)/1e6,clip_spline_ms=(t2-t1)/1e6,multiply_ms=(t3-t2)/1e6,projection_ms=(t4-t3)/1e6,final_shift_ms=(end-t4)/1e6,
            initialization_ms=init_ms,initialization_sent_bytes=init_bytes,online_sent_bytes=ch.sent,online_exchanges=ch.exchanges,
            rounded=rounded,clipped=bq,g=g,weighted_aux=gvec[:,1] if variant=='weighted' else None,raw=raw,acc=acc,out=out,public_classifier=th))
    except BaseException as err:pipe.send(dict(party=p,error=repr(err)))
    finally:sock.close();pipe.close()


def run(a,b,w,spec,variant,rtt=0,gpu_output=None,backend='cpu'):
    assert variant in ('basis','weighted') and backend in ('cpu','cuda')
    a=np.ascontiguousarray(a,dtype=np.int64);b=np.ascontiguousarray(b,dtype=np.uint64);w=np.ascontiguousarray(w,dtype=np.int64)
    n=len(a);m=w.shape[0];view=spec['input_bits']-spec['classification_shift'];shift=spec['fraction_bits']+12
    assert spec['input_bits']==32 and w.shape==(m,n) and np.all(w>=-32768) and np.all(w<=32767)
    assert np.all(a>=-(1<<31)) and np.all(a<(1<<31))
    bound=int(np.abs(w).sum(1).max())*spec['absolute_bound']*32768;assert bound<(1<<62)
    t0=time.perf_counter_ns();sk,alpha,gamma=prepare(n,spec,63,variant);spline_bytes=material_bytes(sk)
    qk=prepare_round(n);ck,_=clip_prepare(n,32);ok=shift_prepare(m,shift);keys=[{},{}]
    for p in (0,1):
        for prefix,source in [('s_',sk[p]),('q_',qk[p]),('c_',ck[p]),('out_',ok[p])]:keys[p].update({prefix+k:v for k,v in source.items()})
    if variant=='basis':
        rg=random64((n,))&MASK;rb=random64((n,))&MASK;rgs=split63(rg);rbs=split63(rb);products=split63(rg*rb)
    mat=random64(w.shape)&MASK;v=random64((n,))&MASK;mats=split63(mat);vs=split63(v);ts=split63(mat@v)
    for p in (0,1):
        keys[p].update(mat_a=mats[p],mat_v=vs[p],mat_product=ts[p])
        if variant=='basis':keys[p].update(mul_rg=rgs[p],mul_rb=rbs[p],mul_product=products[p])
    material=material_bytes(keys);dealer_ms=(time.perf_counter_ns()-t0)/1e6;aa=split63(a.view(np.uint64));bb=split64(b,32);ww=split63(w.view(np.uint64))
    target=worker
    if backend=='cuda':
        from exact_coarse_cuda_protocol import worker_cuda
        target=worker_cuda
    ctx=mp.get_context('spawn');sockets=socket.socketpair();barrier=ctx.Barrier(2);pipes=[];procs=[]
    for p in (0,1):
        parent,child=ctx.Pipe(duplex=False);proc=ctx.Process(target=target,args=(p,sockets[p],(aa[p],bb[p],ww[p]),keys[p],variant,spec,w.shape,shift,rtt,0,barrier,child))
        proc.start();child.close();pipes.append(parent);procs.append(proc)
    for sock in sockets:sock.close()
    rows=[]
    for pipe in pipes:
        if not pipe.poll(90):
            for proc in procs:proc.terminate()
            raise RuntimeError('exact coarse block timeout')
        rows.append(pipe.recv())
    for proc in procs:proc.join(30);assert proc.exitcode==0
    if any('error' in r for r in rows):raise RuntimeError(rows)
    def reconstruct(field):return rows[0].pop(field)+rows[1].pop(field)
    signedb=((b&M32)^np.uint64(1<<31)).astype(np.int64)-(1<<31)
    assert np.array_equal(reconstruct('rounded')&M32,((signedb+128)//256).view(np.uint64)&M32)
    bi=plain_quantize(b,32767);assert np.array_equal(reconstruct('clipped')&MASK,bi.view(np.uint64)&MASK)
    expected_t=(gamma+(a//(1<<spec['classification_shift'])).view(np.uint64))&np.uint64((1<<view)-1)
    assert all(np.array_equal(r.pop('public_classifier'),expected_t) for r in rows)
    gi=plain_spline(spec,a);assert np.array_equal(reconstruct('g')&MASK,gi.view(np.uint64)&MASK)
    if variant=='weighted':
        rb=(sk[0]['b_mask']+sk[1]['b_mask'])&MASK;assert np.array_equal(reconstruct('weighted_aux')&MASK,(-rb*gi.view(np.uint64))&MASK)
    else:
        for r in rows:r.pop('weighted_aux')
    plainraw=gi*bi;assert np.array_equal(reconstruct('raw')&MASK,plainraw.view(np.uint64)&MASK)
    acc=w@plainraw;assert np.array_equal(reconstruct('acc')&MASK,acc.view(np.uint64)&MASK)
    out=reconstruct('out').view(np.int64);assert np.array_equal(out,acc//(1<<shift))
    if gpu_output is not None:assert np.array_equal(out,gpu_output)
    online=sum(r['online_sent_bytes'] for r in rows);assert online==n*(72 if variant=='weighted' else 88)+m*16
    assert all(r['online_exchanges']==5 for r in rows)
    return dict(variant=variant,n=n,m=m,backend=backend,process_context='spawn',classifier_bits=view,low_correction_bits=spec['classification_shift'],emulated_rtt_ms=rtt,
        all_functional_checks_passed=True,gpu_capture_output_checked=gpu_output is not None,parties=rows,
        online_sent_bytes_both_parties=online,total_key_array_bytes_both_parties=material,spline_key_array_bytes_both_parties=spline_bytes,
        model_initialization_sent_bytes_both_parties=sum(r['initialization_sent_bytes'] for r in rows),dealer_ms=dealer_ms,
        all_domain_integer_accumulator_bound=bound,functionality='Same deterministic fine aligned function as exact32; no classifier approximation.')
