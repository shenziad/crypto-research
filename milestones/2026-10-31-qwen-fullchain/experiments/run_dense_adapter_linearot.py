"""Actual dense shared-score baseline component; not full private sampling."""
import json
from pathlib import Path
import secrets
import socket

import numpy as np
from bounded_adapter_fixed16 import encode_parameters,tanh_integer,SCALE
import run_fable_smallbatch_check as transport
from run_bounded_adapter_gc import fixtures as old_fixtures

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return transport.sha(p)


def port_triple():
    for _ in range(100):
        base=secrets.randbelow(25000)+30000;sockets=[]
        try:
            for delta in (0,50,100):
                s=socket.socket();sockets.append(s);s.bind(('127.0.0.1',base+delta))
            return base
        except OSError:pass
        finally:
            for s in sockets:s.close()
    raise RuntimeError('could not find three public local ports')


def fixtures():
    for name,mode,x,b,a,indices in old_fixtures():
        if name in ('signed_pwl_knots','near_domain_accumulators'):
            yield name,x,b,a
    with np.load(ROOT/'results/bounded_output_adapter_weights.npz') as w:
        a,b=encode_parameters(w['bounded_A'],w['bounded_B'])
    with np.load(ROOT/'results/bounded_adapter_gc_inputs.npz') as w:xs=w['x']
    for i,corpus in enumerate(('ptb','wiki')):
        yield corpus+'_full50304',np.rint(xs[i]*SCALE).astype(np.int64),b,a


def input_bytes(x,b,a,role):
    if role==1:
        x=np.zeros_like(x)
    else:
        b=np.zeros_like(b);a=np.zeros_like(a)
    arrays=[(x.astype(np.int64)&((1<<23)-1)).astype('<u8'),
        (b.astype(np.int64)&((1<<18)-1)).astype('<u8'),
        (a.astype(np.int64)&((1<<18)-1)).astype('<u8')]
    return b''.join(z.tobytes() for z in arrays)


def reconstruct(paths,name,width):
    a,b=[np.frombuffer((p/name).read_bytes(),dtype='<u8') for p in paths]
    v=((a+b)&((1<<width)-1)).astype(np.int64)
    return np.where(v&(1<<(width-1)),v-(1<<width),v)


def main():
    registration=ROOT/'notes/20261030_dense_adapter_linearot_preregistration.json'
    spec=json.loads(registration.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/dense_adapter_linearot.json';assert not destination.exists()
    folder=ROOT/'results/dense_adapter_linearot_runs';folder.mkdir()
    records=[]
    for name,x,b,a in fixtures():
        assert np.linalg.norm(x.astype(float)/SCALE)<=20
        assert np.abs(a).max()<=SCALE and np.abs(b).max()<=SCALE
        raw_t=x@b.T;t=raw_t//SCALE;raw_z=a@t;delta=tanh_integer(raw_z//SCALE)
        sub=folder/name;sub.mkdir();paths=[sub/f'party{r}' for r in (1,2)]
        for p in paths:p.mkdir()
        inputs=[p/'own_input.bin' for p in paths]
        for role,p in enumerate(inputs,1):p.write_bytes(input_bytes(x,b,a,role))
        port=port_triple();binary=ROOT/'toolchain/bounded-sampler/dense_adapter_linearot'
        commands=[[str(binary),str(role),str(port),str(len(a)),str(inputs[role-1]),str(paths[role-1]),'1'] for role in (1,2)]
        record=transport.pair(commands,sub,timeout=600)
        for filename,width,expected in [('raw_t_add41.bin',41,raw_t),('t_add26.bin',26,t),
            ('raw_z_add44.bin',44,raw_z),('delta_add44.bin',44,delta)]:
            assert np.array_equal(reconstruct(paths,filename,width),expected),(name,filename)
        party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        for channel in range(3):
            assert party[0]['end_counters'][channel]==party[1]['end_counters'][channel+3]
            assert party[1]['end_counters'][channel]==party[0]['end_counters'][channel+3]
        keys=['setup_counters','bx_counters','bridge_counters','az_counters','end_counters']
        stage_bytes={};previous=0
        for key,stage in zip(keys,['setup','Bx','floor_bridge','A_t','floor_tanh_share']):
            value=sum(sum(p[key][:3]) for p in party);stage_bytes[stage]=value-previous;previous=value
        gates=[json.loads((p/'nonlinear_gate_count.json').read_bytes()) for p in paths]
        assert gates[0]==gates[1]
        record.update(name=name,rows=len(a),all_correct=True,parties=party,stage_wire_bytes=stage_bytes,
            total_wire_bytes=previous,max_party_protocol_ms=max(p['protocol_ms'] for p in party),
            shared_scores_exact=True,exact_intermediates_checked=16+2*len(a),nonlinear_gates=gates[0],
            file_sha256={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(record)
        print(name,'allsharedscores exact, bytes',previous,'protocolms',record['max_party_protocol_ms'],stage_bytes,flush=True)
    out=dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],
        preregistration_sha256=sha(registration),limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,actual_pairs=len(records)),indent=2))


if __name__=='__main__':main()
