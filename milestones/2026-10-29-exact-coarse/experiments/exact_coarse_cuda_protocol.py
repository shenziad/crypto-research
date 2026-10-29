"""CUDA exact aligned classifier/body block; explicit low-carry correction."""
import time
import numpy as np
import torch
from dcf_cuda import select,contract
from public_spline_cuda import prepare_runtime,evaluate
from public_spline_cuda_protocol import upload
from exact_coarse_cuda import compile_public
from public_spline_fss import queries as clip_queries
from public_spline_contraction_protocol import PayloadDelayChannel,clip_spec

MASK=(1<<63)-1;M32=(1<<32)-1


def worker_cuda(p,sock,inputs,key,variant,spec,shape,shift,rtt,bandwidth,barrier,pipe):
    try:
        setup=time.perf_counter_ns();torch.set_num_threads(1);torch.cuda.set_device(0)
        a,b,w=[upload(v) for v in inputs];n=len(a);ch=PayloadDelayChannel(sock,p,rtt);view=spec['input_bits']-spec['classification_shift'];vm=(1<<view)-1;lowbits=spec['classification_shift']
        sk_cpu={k[2:]:v for k,v in key.items() if k.startswith('s_')};ak={k:upload(v) for k,v in sk_cpu.items() if k.startswith('classifier_')}
        rt=prepare_runtime({k:v for k,v in sk_cpu.items() if not k.startswith('classifier_')},np.zeros(n,dtype=np.uint64),spec,variant);sk=rt['key']
        key={k:upload(v) for k,v in key.items() if not k.startswith('s_')};ck={k[2:]:v for k,v in key.items() if k.startswith('c_')};qk={k[2:]:v for k,v in key.items() if k.startswith('q_')}
        coords={d:upload(np.arange(d,dtype=np.int32)) for d in (1,2,4)}
        def dcf(k,x,bits,dim=1,group=64):return select(k,x.reshape(-1,1).contiguous(),coords[dim],p,bits,dim,group,True)[:,0,:]
        def clip_eval(host):
            q,weights,constant,ac=clip_queries(clip_spec(),host,32);values=select(ck['dcf'],upload(q),coords[2],p,32,2,64,True)
            out=(values*upload(weights)).sum((1,2))+upload(ac)*ck['alpha_share']+ck['refresh']
            if p==0:out=out+upload(constant)
            return out&MASK
        def swap(arrays,widths):
            host=[v.detach().cpu().numpy().view(np.uint64) for v in arrays];packet=b''.join(v.astype('<u4' if width==4 else '<u8').tobytes() for v,width in zip(host,widths));peer=ch.swap(packet);offset=0;result=[]
            for v,width in zip(host,widths):
                other=np.frombuffer(peer,dtype='<u4' if width==4 else '<u8',count=v.size,offset=offset).reshape(v.shape).astype(np.uint64)
                opened=(v+other)&np.uint64(M32 if width==4 else MASK);offset+=v.size*width;result.append((opened,upload(opened)))
            assert offset==len(peer)
            return result
        zero=torch.zeros(n,device='cuda',dtype=torch.int64)
        evaluate(rt,p);dcf(ak['classifier_low'],zero,lowbits,1,32);dcf(qk['q_low'],zero,8,1,32);dcf(qk['q_high'],zero,32,1,32)
        clip_eval(np.zeros(n,dtype=np.uint64));dcf(key['out_low'],zero[:shape[0]],shift);dcf(key['out_high'],zero[:shape[0]],63)
        contract(key['mat_a'],key['mat_a'],key['mat_v'],zero,key['mat_product'],p);torch.cuda.synchronize();setup_ms=(time.perf_counter_ns()-setup)/1e6
        start=time.perf_counter_ns();wh=swap([(w+key['mat_a'])&MASK],[8])[0][1];torch.cuda.synchronize();init_ms=(time.perf_counter_ns()-start)/1e6;init_bytes=ch.sent;ch.sent=0;ch.exchanges=0;ch.bandwidth_mbps=bandwidth
        barrier.wait();start=time.perf_counter_ns()
        (_,ah),(_,bh)=swap([(a+sk['input_mask'])&MASK,(b+qk['q_mask'])&M32],[8,4])
        rounded=qk['q_const']+dcf(qk['q_low'],bh&255,8,1,32)[:,0]+dcf(qk['q_high'],bh,32,1,32)[:,0]
        aq=ak['classifier_constant']+dcf(ak['classifier_low'],ah&((1<<lowbits)-1),lowbits,1,32)[:,0]
        if p==0:rounded=rounded+(bh>>8);aq=aq+((ah&M32)>>lowbits)
        rounded=rounded&M32;aq=aq&vm;torch.cuda.synchronize();t1=time.perf_counter_ns()
        (chb_host,_),(th_host,th)=swap([(rounded+ck['input_mask'])&M32,aq],[4,4]);th=th&vm;th_host=th_host&np.uint64(vm)
        bq=clip_eval(chb_host);compile_public(rt,ah,th);values=evaluate(rt,p);gvec=values if variant=='weighted' else None;g=values[:,0] if variant=='weighted' else values
        torch.cuda.synchronize();t2=time.perf_counter_ns();bm=(bq+(sk['b_mask'] if variant=='weighted' else key['mul_rb']))&MASK
        if variant=='weighted':(_,bh)=swap([bm],[8])[0];raw=bh*gvec[:,0]+gvec[:,1]
        else:
            (_,bh),(_,gh)=swap([bm,(g+key['mul_rg'])&MASK],[8,8]);raw=key['mul_product']-gh*key['mul_rb']-bh*key['mul_rg']
            if p==0:raw=raw+gh*bh
        raw=raw&MASK;torch.cuda.synchronize();t3=time.perf_counter_ns();zh=swap([(raw+key['mat_v'])&MASK],[8])[0][1]
        acc=contract(wh,key['mat_a'],key['mat_v'],zh,key['mat_product'],p);torch.cuda.synchronize();t4=time.perf_counter_ns();sh=swap([(acc+key['out_mask'])&MASK],[8])[0][1]
        out=key['out_constant']+dcf(key['out_low'],sh&((1<<shift)-1),shift)[:,0]+dcf(key['out_high'],sh,63)[:,0]
        if p==0:out=out+(sh>>shift)
        torch.cuda.synchronize();end=time.perf_counter_ns()
        def host(v):return v.detach().cpu().numpy().view(np.uint64)
        pipe.send(dict(party=p,total_ms=(end-start)/1e6,round1_ms=(t1-start)/1e6,clip_spline_ms=(t2-t1)/1e6,multiply_ms=(t3-t2)/1e6,projection_ms=(t4-t3)/1e6,final_shift_ms=(end-t4)/1e6,
            cuda_setup_upload_ms=setup_ms,initialization_ms=init_ms,initialization_sent_bytes=init_bytes,online_sent_bytes=ch.sent,online_exchanges=ch.exchanges,
            rounded=host(rounded),clipped=host(bq),g=host(g),weighted_aux=host(gvec[:,1]) if variant=='weighted' else None,raw=host(raw),acc=host(acc),out=host(out),public_classifier=th_host))
    except BaseException as err:pipe.send(dict(party=p,error=repr(err)))
    finally:sock.close();pipe.close()
