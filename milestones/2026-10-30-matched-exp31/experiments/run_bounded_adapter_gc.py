"""Private finite arithmetic + acceptance conformance on model/boundary inputs."""
import json
import os
from pathlib import Path
import secrets

import numpy as np

from bounded_adapter_fixed16 import SCALE,encode_parameters,tanh_integer
from finite_descending_bernoulli import acceptance,first_success
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return transport.sha(path)


def fixtures():
    # Cover each quarter-width tanh knot from both sides and both signs.
    x=np.zeros(128,dtype=np.int64);x[0]=8*SCALE
    b=np.zeros((8,128),dtype=np.int64);b[0,0]=SCALE
    targets=np.array([j*16384+offset for j in range(-16,17) for offset in (-8,0,8)],dtype=np.int64)
    a=np.zeros((len(targets),8),dtype=np.int64);a[:,0]=targets//8
    yield 'signed_pwl_knots','shares',x,b,a,np.arange(len(a),dtype=np.int64)
    x=np.full(128,7*SCALE//4,dtype=np.int64)
    b=np.full((8,128),SCALE,dtype=np.int64);b[4:]*=-1
    a=np.tile(np.array([1,1,1,1,-1,-1,-1,-1],dtype=np.int64),(8,1))*SCALE
    a[1::2]*=-1
    yield 'near_domain_accumulators','shares',x,b,a,np.arange(8,dtype=np.int64)
    with np.load(ROOT/'results/bounded_output_adapter_weights.npz') as weights:
        a,b=encode_parameters(weights['bounded_A'],weights['bounded_B'])
    with np.load(ROOT/'results/bounded_adapter_gc_inputs.npz') as captures:
        xs=captures['x'];qs=captures['q']
    rng=np.random.default_rng(20261030)
    for i,corpus in enumerate(('ptb','wiki')):
        x=np.rint(xs[i]*SCALE).astype(np.int64)
        indices=rng.choice(50304,201,p=qs[i]/qs[i].sum())
        for mode in ('shares','token'):
            yield corpus+'_201_'+mode,mode,x,b,a[indices],indices


def main():
    registration=ROOT/'notes/20261030_bounded_adapter_gc_preregistration_v2.json'
    spec=json.loads(registration.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/bounded_adapter_gc.json';assert not destination.exists()
    folder=ROOT/'results/bounded_adapter_gc_runs';folder.mkdir()
    binary=ROOT/'toolchain/bounded-sampler/bounded_adapter_gc'
    records=[]
    for name,mode,x,b,a,indices in fixtures():
        assert np.linalg.norm(x.astype(float)/SCALE)<=20.001
        assert np.abs(a).max()<=SCALE and np.abs(b).max()<=SCALE
        t=(x@b.T)//SCALE
        z=(a@t)//SCALE
        delta=tanh_integer(z)
        sub=folder/name;sub.mkdir()
        data=[bytearray(),bytearray()]
        # Client knows x, server knows B. Their unused shares are zero.
        for value in x:
            data[0].extend((0).to_bytes(4,'little'))
            data[1].extend((int(value)&((1<<22)-1)).to_bytes(4,'little'))
        for value in b.ravel():
            data[0].extend((int(value)&((1<<18)-1)).to_bytes(4,'little'))
            data[1].extend((0).to_bytes(4,'little'))
        accepted=[]
        for row,index,score in zip(a,indices,delta):
            coins=[[secrets.randbits(44) for _ in range(19)] for _ in range(2)]
            accepted.append(acceptance(int(score),coins)[0])
            words=[(int(v)&((1<<18)-1),18,4) for v in row]
            words += [(int(index),16,4)]
            words += [(v,44,8) for chain in coins for v in chain]
            for value,width,size in words:
                mask=secrets.randbits(width)
                data[0].extend(mask.to_bytes(size,'little'))
                data[1].extend((value^mask).to_bytes(size,'little'))
        inputs=[sub/f'party{i+1}_input.bin' for i in range(2)]
        outputs=[sub/f'party{i+1}_output.bin' for i in range(2)]
        metrics=[sub/f'party{i+1}_metrics.json' for i in range(2)]
        for path,value in zip(inputs,data):path.write_bytes(value)
        port=transport.port()
        commands=[[str(binary),str(i+1),str(port),str(len(a)),str(inputs[i]),str(outputs[i]),str(metrics[i]),mode]
                  for i in range(2)]
        record=transport.pair(commands,sub,timeout=180)
        if mode=='shares':
            shares=[np.frombuffer(p.read_bytes(),dtype='<u4') for p in outputs]
            actual=(shares[0]^shares[1]).view('<i4').astype(np.int64)
            expected=np.concatenate((t,z,delta))
            assert np.array_equal(actual,expected),(name,actual[:8],expected[:8])
            record['exact_intermediate_values_checked']=len(expected)
        else:
            assert outputs[0].read_bytes()==b''
            expected=first_success(indices.tolist(),accepted,65536)
            assert int(outputs[1].read_text())==expected
            record['expected_final_token']=expected
        parties=[json.loads(p.read_bytes()) for p in metrics]
        assert parties[0]['sent']==parties[1]['received'] and parties[1]['sent']==parties[0]['received']
        assert parties[0]['and_gates_before_release']==parties[1]['and_gates_before_release']
        record.update(name=name,mode=mode,count=len(a),all_correct=True,parties=parties,
            total_wire_bytes=sum(p['sent'] for p in parties),
            max_party_protocol_ms=max(p['protocol_ms'] for p in parties),
            input_share_sha256=[sha(p) for p in inputs],output_sha256=[sha(p) for p in outputs],
            max_abs_t_integer=int(np.abs(t).max()),max_abs_z_integer=int(np.abs(z).max()),
            max_abs_delta_integer=int(np.abs(delta).max()))
        records.append(record)
        print(name,'correct, bytes',record['total_wire_bytes'],'ms',record['max_party_protocol_ms'],flush=True)
    out=dict(all_passed=True,results=records,preregistration_sha256=sha(registration),
        input_sha256=spec['input_sha256'],limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,actual_two_party_pairs=len(records)),indent=2))


if __name__=='__main__':main()
