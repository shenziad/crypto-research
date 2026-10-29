"""Spawned CUDA party for the full finite gate/quantize/private-down block.

Input/key uploads and model initialization are separately recorded. Online
wall time includes public query compilation, kernels, host/device transfers,
socket exchanges and synchronization. One physical GPU serves both processes.
"""
import time
import numpy as np
import torch
from dcf_cuda import tensor,select,contract
from public_spline_cuda import prepare_runtime,compile_public,evaluate
from public_spline_fss import queries as clip_queries
from public_spline_contraction_protocol import PayloadDelayChannel,clip_spec

MASK=(1<<63)-1;M32=(1<<32)-1


def upload(v):
    if v.dtype==np.uint32:v=v.astype(np.int64)
    return tensor(v)


def worker_cuda(p,sock,inputs,key,variant,spec,shape,shift,rtt,bandwidth_mbps,barrier,pipe):
    try:
        setup_start=time.perf_counter_ns();torch.set_num_threads(1);torch.cuda.set_device(0)
        a,b,w=[upload(v) for v in inputs];n=len(a);ch=PayloadDelayChannel(sock,p,rtt)
        sk_cpu={k[2:]:v for k,v in key.items() if k.startswith('s_')}
        rt=prepare_runtime(sk_cpu,np.zeros(n,dtype=np.uint64),spec,variant);sk=rt['key']
        key={k:upload(v) for k,v in key.items() if not k.startswith('s_')}
        ck={k[2:]:v for k,v in key.items() if k.startswith('c_')}
        qk={k[2:]:v for k,v in key.items() if k.startswith('q_')}
        coords={d:upload(np.arange(d,dtype=np.int32)) for d in (1,2,4)}
        def dcf(k,x,bits,dim=1,group=64):return select(k,x.reshape(-1,1).contiguous(),coords[dim],p,bits,dim,group,True)[:,0,:]
        def clip_eval(opened_host):
            q,weights,constant,ac=clip_queries(clip_spec(),opened_host,32)
            values=select(ck['dcf'],upload(q),coords[2],p,32,2,64,True)
            out=(values*upload(weights)).sum((1,2))+upload(ac)*ck['alpha_share']+ck['refresh']
            if p==0:out=out+upload(constant)
            return out&MASK
        def swap_arrays(arrays,widths):
            host=[v.detach().cpu().numpy().view(np.uint64) for v in arrays]
            packet=b''.join(v.astype('<u4' if width==4 else '<u8').tobytes() for v,width in zip(host,widths));peer=ch.swap(packet)
            result=[];offset=0
            for v,width in zip(host,widths):
                other=np.frombuffer(peer,dtype='<u4' if width==4 else '<u8',count=v.size,offset=offset).reshape(v.shape).astype(np.uint64)
                opened=(v+other)&np.uint64(M32 if width==4 else MASK);offset+=v.size*width
                result.append((opened,upload(opened)))
            assert offset==len(peer)
            return result
        # Eagerly load CUDA modules and execute local dummy evaluations. These
        # values are discarded, never sent or reconstructed. FSS key holders
        # may locally evaluate their key; actual protocol openings occur once.
        zero=torch.zeros(n,device='cuda',dtype=torch.int64)
        evaluate(rt,p);dcf(qk['q_low'],zero,8,1,32);dcf(qk['q_high'],zero,32,1,32)
        clip_eval(np.zeros(n,dtype=np.uint64));dcf(key['out_low'],zero[:shape[0]],shift);dcf(key['out_high'],zero[:shape[0]],63)
        contract(key['mat_a'],key['mat_a'],key['mat_v'],zero,key['mat_product'],p)
        torch.cuda.synchronize();setup_ms=(time.perf_counter_ns()-setup_start)/1e6
        start=time.perf_counter_ns();public_w=swap_arrays([(w+key['mat_a'])&MASK],[8])[0][1]
        torch.cuda.synchronize();init_ms=(time.perf_counter_ns()-start)/1e6;init_bytes=ch.sent;ch.sent=0;ch.exchanges=0
        ch.bandwidth_mbps=bandwidth_mbps;barrier.wait();start=time.perf_counter_ns()
        (ah_host,ah),(bh_host,bh)=swap_arrays([(a+sk['input_mask'])&MASK,(b+qk['q_mask'])&M32],[8,4])
        rounded=qk['q_const']+dcf(qk['q_low'],bh&255,8,1,32)[:,0]+dcf(qk['q_high'],bh,32,1,32)[:,0]
        if p==0:rounded=rounded+(bh>>8)
        rounded=rounded&M32;compile_public(rt,ah);coefficients=None;gvec=None
        values=evaluate(rt,p)
        if variant=='selector':coefficients=values
        elif variant=='weighted':gvec=values;g=gvec[:,0]
        else:g=values
        torch.cuda.synchronize();t1=time.perf_counter_ns()
        cm=(rounded+ck['input_mask'])&M32
        if variant=='weighted':parts=swap_arrays([cm],[4])
        else:
            gm=((coefficients[:,0]+sk['horner_rs']) if variant=='selector' else (g+key['mul_rg']))&MASK
            parts=swap_arrays([cm,gm],[4,8]);gh=parts[1][1]
        chb_host,chb=parts[0]
        bq=clip_eval(chb_host)
        if variant=='selector':
            g=sk['horner_product']-ah*sk['horner_rs']-gh*sk['input_mask']+coefficients[:,1]+sk['refresh']
            if p==0:g=g+ah*gh
            g=g&MASK
        torch.cuda.synchronize();t2=time.perf_counter_ns()
        bm=(bq+(sk['b_mask'] if variant=='weighted' else key['mul_rb']))&MASK
        if variant=='selector':
            (_,bh),(_,gh)=swap_arrays([bm,(g+key['mul_rg'])&MASK],[8,8])
        else:(_,bh)=swap_arrays([bm],[8])[0]
        if variant=='weighted':raw=bh*gvec[:,0]+gvec[:,1]
        else:
            raw=key['mul_product']-gh*key['mul_rb']-bh*key['mul_rg']
            if p==0:raw=raw+gh*bh
        raw=raw&MASK;torch.cuda.synchronize();t3=time.perf_counter_ns()
        zh=swap_arrays([(raw+key['mat_v'])&MASK],[8])[0][1]
        acc=contract(public_w,key['mat_a'],key['mat_v'],zh,key['mat_product'],p)
        torch.cuda.synchronize();t4=time.perf_counter_ns()
        sh=swap_arrays([(acc+key['out_mask'])&MASK],[8])[0][1]
        out=key['out_constant']+dcf(key['out_low'],sh&((1<<shift)-1),shift)[:,0]+dcf(key['out_high'],sh,63)[:,0]
        if p==0:out=out+(sh>>shift)
        torch.cuda.synchronize();end=time.perf_counter_ns()
        def host(v):return v.detach().cpu().numpy().view(np.uint64)
        pipe.send(dict(party=p,total_ms=(end-start)/1e6,first_round_spline_ms=(t1-start)/1e6,
            clip_ms=(t2-t1)/1e6,multiply_ms=(t3-t2)/1e6,projection_ms=(t4-t3)/1e6,final_shift_ms=(end-t4)/1e6,
            cuda_setup_upload_ms=setup_ms,cuda_setup_includes='Context, key/input uploads, public descriptor compilation and discarded local warmup evaluations; no protocol openings during warmup.',initialization_ms=init_ms,initialization_sent_bytes=init_bytes,
            online_sent_bytes=ch.sent,online_exchanges=ch.exchanges,
            rounded=host(rounded),clipped=host(bq),g=host(g),weighted_aux=host(gvec[:,1]) if variant=='weighted' else None,
            raw=host(raw),acc=host(acc),out=host(out)))
    except BaseException as err:pipe.send(dict(party=p,error=repr(err)))
    finally:sock.close();pipe.close()
