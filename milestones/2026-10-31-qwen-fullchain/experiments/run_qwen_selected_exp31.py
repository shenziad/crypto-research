"""At most10 native pairs: Qwen896, XOR18 indices, selected rows and E31.

Synthetic/offline share fixtures test arithmetic only. No private LUT chain or
full Qwen sampler is claimed. Parent reads output files only after both exit.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets
import time

import numpy as np
from qwen_bounded_fixed16 import encode_parameters
from run_selected_adapter_exp31 import expected
from run_dense_adapter_linearot import port_triple,reconstruct,sha
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]
S=65536;FAIL=262144


def fixtures():
    with np.load(ROOT/'results/bounded_adapter_qwen_transfer_weights.npz') as z:
        a,b=encode_parameters(z['bounded_A'],z['bounded_B'])
    with np.load(ROOT/'results/bounded_adapter_qwen_frozen_captures.npz') as cap:
        for j,corpus in enumerate(('ptb_test','wiki_test')):
            x=np.rint(cap[corpus+'_x'].astype(np.float64)*S).astype(np.int64)
            indices=np.random.default_rng(20261030+j).integers(0,151936,201,dtype=np.int64)
            indices[:4]=[0,65535,65536,151935]
            yield corpus+'_201',x,b,a[indices],indices,'random'
    for mode in ('early','late','failure'):
        x=np.zeros(896,dtype=np.int64);x[0]=S
        b=np.zeros((8,896),dtype=np.int64)
        indices=np.arange(201,dtype=np.int64)+65536;indices[-1]=151935
        yield mode+'_201',x,b,np.zeros((201,8),dtype=np.int64),indices,mode
    # Coordinates outside old signed22 domain, all within new L2<=52 domain.
    for sign in (1,-1):
        x=np.zeros(896,dtype=np.int64);x[895]=sign*52*S
        b=np.zeros((8,896),dtype=np.int64);b[0,895]=S
        a=np.zeros((1,8),dtype=np.int64);a[0,0]=S//2
        yield ('positive' if sign==1 else 'negative')+'_x52',x,b,a,np.array([151935]),'before'
    x=np.zeros(896,dtype=np.int64);x[0]=8*S
    b=np.zeros((8,896),dtype=np.int64);b[0,0]=S
    for name,coefficient,mode in [('positive_endpoint',S//2,'maximum'),
                                  ('negative_equal',-S//2,'equal'),
                                  ('negative_before',-S//2,'before')]:
        a=np.zeros((1,8),dtype=np.int64);a[0,0]=coefficient
        yield name,x,b,a,np.array([151935]),mode


def audit(paths,x,b,a,indices,uniform):
    rt,t,rz,delta,e=expected(x,b,a)
    for name,width,value in [('a_add18.bin',18,a.ravel()),('raw_t_add41.bin',41,rt),
                           ('t_add26.bin',26,t),('raw_z_add44.bin',44,rz),
                           ('delta_add44.bin',44,delta),('exp_add33.bin',33,e)]:
        for p in paths:assert (p/name).stat().st_size==8*value.size
        assert np.array_equal(reconstruct(paths,name,width),value),name
    accept=np.asarray(uniform,dtype=np.int64)<e
    token=next((int(i) for i,ok in zip(indices,accept) if ok),FAIL)
    assert (paths[0]/'final_output.txt').read_bytes()==b''
    assert int((paths[1]/'final_output.txt').read_text())==token
    return dict(all_correct=True,rows=len(a),input_dimension=896,
        index_bits=18,finite_scores_and_E31_exact=True,token_matches=True,
        failure=token==FAIL,high_token_exercised=token>65535 and token!=FAIL)


def main():
    reg=ROOT/'notes/20261030_qwen_selected_exp31_preregistration.json'
    spec=json.loads(reg.read_bytes())
    for n,h in spec['input_sha256'].items():assert sha(ROOT/n)==h,n
    destination=ROOT/'results/qwen_selected_exp31.json';assert not destination.exists()
    folder=ROOT/'results/qwen_selected_exp31_runs';folder.mkdir();rows=[]
    started=time.monotonic()
    budget_start=datetime.fromisoformat(spec['stage_budget_started_utc'])
    for name,x,b,a,indices,mode in fixtures():
        remaining=1800-(datetime.now(timezone.utc)-budget_start).total_seconds()
        assert remaining>0,'implementation-stage30min budget reached'
        e=expected(x,b,a)[-1];n=len(a)
        if mode=='random':u=np.array([secrets.randbits(31) for _ in range(n)],dtype=np.int64)
        elif mode in ('early','late','failure'):
            u=np.full(n,(1<<31)-1,dtype=np.int64)
            if mode=='early':u[0]=0
            if mode=='late':u[-1]=0
        elif mode=='maximum':u=np.full(n,(1<<31)-1,dtype=np.int64)
        elif mode=='equal':u=e.copy()
        else:u=e-1
        sub=folder/name;sub.mkdir();paths=[sub/f'party{i}' for i in (1,2)]
        data=[bytearray(),bytearray()]
        for v in x:
            data[0].extend((0).to_bytes(4,'little'));data[1].extend((int(v)&((1<<23)-1)).to_bytes(4,'little'))
        for v in b.ravel():
            data[0].extend((int(v)&((1<<18)-1)).to_bytes(4,'little'));data[1].extend((0).to_bytes(4,'little'))
        for row,index,uniform in zip(a,indices,u):
            for coefficient in row:
                mask=secrets.randbits(18);value=int(coefficient)&((1<<18)-1)
                data[0].extend(mask.to_bytes(4,'little'));data[1].extend((mask^value).to_bytes(4,'little'))
            mask=secrets.randbits(18)
            data[0].extend(mask.to_bytes(4,'little'));data[1].extend((mask^int(index)).to_bytes(4,'little'))
            for j in range(38):
                first=secrets.randbits(44);second=first^int(uniform) if j==0 else secrets.randbits(44)
                data[0].extend(first.to_bytes(8,'little'));data[1].extend(second.to_bytes(8,'little'))
        for p,value in zip(paths,data):p.mkdir();(p/'own_input.bin').write_bytes(value)
        port=port_triple();binary=ROOT/'toolchain/bounded-sampler/qwen_selected_exp31'
        cmd=[[str(binary),str(j+1),str(port),str(n),str(p/'own_input.bin'),
              str(p/'final_output.txt'),str(p/'metrics.json'),'token'] for j,p in enumerate(paths)]
        row=transport.pair(cmd,sub,timeout=min(180,remaining));row.update(audit(paths,x,b,a,indices,u))
        parties=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert parties[0]['sent']==parties[1]['received'] and parties[1]['sent']==parties[0]['received']
        row.update(name=name,parties=parties,total_wire_bytes=sum(p['sent'] for p in parties),
            files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        rows.append(row);print(name,'native Qwen head correct',row['total_wire_bytes'],flush=True)
    assert len(rows)==10
    out=dict(all_passed=True,results=rows,preregistration_sha256=sha(reg),
        input_sha256=spec['input_sha256'],elapsed_seconds=time.monotonic()-started,limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print('all10 native pairs passed',flush=True)


if __name__=='__main__':main()
