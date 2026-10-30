"""After-exit conformance for the same-E31 secret selected-row head."""
import json
from pathlib import Path
import secrets

import numpy as np
from bounded_adapter_fixed16 import tanh_integer,SCALE
from run_bounded_adapter_gc import fixtures as old_fixtures
from run_dense_adapter_linearot import port_triple,reconstruct,sha
from run_dense_adapter_cdf31 import exp31
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def expected(x,b,a):
    raw_t=x@b.T;t=raw_t//SCALE;raw_z=a@t;delta=tanh_integer(raw_z//SCALE)
    cert=json.loads((ROOT/'results/sci_exp31_finite_certificate.json').read_bytes())
    e=exp31(delta,cert['actual_public_factors']['tables'])
    return raw_t,t,raw_z,delta,e


def audit_head(paths,x,b,a,indices,uniform):
    raw_t,t,raw_z,delta,e=expected(x,b,a)
    for name,width,value in [('a_add18.bin',18,a.ravel()),('raw_t_add41.bin',41,raw_t),
        ('t_add26.bin',26,t),('raw_z_add44.bin',44,raw_z),('delta_add44.bin',44,delta),('exp_add33.bin',33,e)]:
        assert np.array_equal(reconstruct(paths,name,width),value),name
    accepted=np.asarray(uniform,dtype=np.int64)<e
    selected=next((int(i) for i,ok in zip(indices,accepted) if ok),65536)
    assert (paths[0]/'final_output.txt').read_bytes()==b''
    assert int((paths[1]/'final_output.txt').read_text())==selected
    return dict(all_correct=True,rows=len(a),shared_A_conversion_exact=True,
        all_scores_and_E31_exact=True,final_token_matches=True,failure=selected==65536)


def fixtures():
    for name,mode,x,b,a,indices in old_fixtures():
        if mode=='shares':yield name,x,b,a,indices,'random'
    x=np.zeros(128,dtype=np.int64);x[0]=SCALE;b=np.zeros((8,128),dtype=np.int64)
    for mode in ('early','late','failure'):
        yield mode+'_201',x,b,np.zeros((201,8),dtype=np.int64),np.arange(201),mode
    x=np.zeros(128,dtype=np.int64);x[0]=8*SCALE
    b=np.zeros((8,128),dtype=np.int64);b[0,0]=SCALE
    for name,coefficient,mode in [('positive_endpoint',SCALE,'maximum'),('negative_equal',-SCALE,'equal'),('negative_before',-SCALE,'before')]:
        a=np.zeros((1,8),dtype=np.int64);a[0,0]=coefficient
        yield name,x,b,a,np.array([50303]),mode


def main():
    reg=ROOT/'notes/20261030_selected_adapter_exp31_preregistration.json';spec=json.loads(reg.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/selected_adapter_exp31.json';assert not destination.exists()
    folder=ROOT/'results/selected_adapter_exp31_runs';folder.mkdir();records=[]
    for name,x,b,a,indices,mode in fixtures():
        e=expected(x,b,a)[-1];count=len(a)
        if mode=='random':u=np.array([secrets.randbits(31) for _ in range(count)],dtype=np.int64)
        elif mode in ('early','late','failure'):
            u=np.full(count,(1<<31)-1,dtype=np.int64)
            if mode=='early':u[0]=0
            if mode=='late':u[-1]=0
        elif mode=='maximum':u=np.array([(1<<31)-1])
        elif mode=='equal':u=e.copy()
        else:u=e-1
        sub=folder/name;sub.mkdir();paths=[sub/f'party{i}' for i in (1,2)]
        data=[bytearray(),bytearray()]
        for value in x:
            data[0].extend((0).to_bytes(4,'little'));data[1].extend((int(value)&((1<<22)-1)).to_bytes(4,'little'))
        for value in b.ravel():
            data[0].extend((int(value)&((1<<18)-1)).to_bytes(4,'little'));data[1].extend((0).to_bytes(4,'little'))
        for row,index,uniform in zip(a,indices,u):
            for coefficient in row:
                mask=secrets.randbits(18);value=int(coefficient)&((1<<18)-1)
                data[0].extend(mask.to_bytes(4,'little'));data[1].extend((value^mask).to_bytes(4,'little'))
            mask=secrets.randbits(16);data[0].extend(mask.to_bytes(4,'little'));data[1].extend((mask^int(index)).to_bytes(4,'little'))
            for j in range(38):
                first=secrets.randbits(44);second=(first^int(uniform)) if j==0 else secrets.randbits(44)
                data[0].extend(first.to_bytes(8,'little'));data[1].extend(second.to_bytes(8,'little'))
        for p,value in zip(paths,data):p.mkdir();(p/'own_input.bin').write_bytes(value)
        port=port_triple();binary=ROOT/'toolchain/bounded-sampler/selected_adapter_exp31'
        commands=[[str(binary),str(i+1),str(port),str(count),str(p/'own_input.bin'),str(p/'final_output.txt'),str(p/'metrics.json'),'token'] for i,p in enumerate(paths)]
        row=transport.pair(commands,sub,timeout=180);row.update(audit_head(paths,x,b,a,indices,u))
        party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert party[0]['sent']==party[1]['received'] and party[1]['sent']==party[0]['received']
        row.update(name=name,total_wire_bytes=sum(p['sent'] for p in party),parties=party,
            files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(row);print(name,'matchedE31 head correct, bytes',row['total_wire_bytes'],flush=True)
    out=dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],preregistration_sha256=sha(reg),limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_passed=True,pairs=len(records))))


if __name__=='__main__':main()
