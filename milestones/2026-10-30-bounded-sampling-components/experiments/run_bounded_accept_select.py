"""Actual two-party GC acceptance/select conformance; no private score producer."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct

from finite_descending_bernoulli import acceptance, first_success, threshold_from_delta
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]
S=1 << 44
Q=1 << 20
L=19


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_chain(delta, mode):
    m=threshold_from_delta(delta)
    if mode=='accept':
        if m<S:
            return [m]+[0]*(L-1)
        return [S-1,S-2,S-2]+[0]*(L-3)
    if mode=='reject':
        assert m>0
        return [0]*L
    if mode=='overflow':
        assert m>=L
        return [m-1-j for j in range(L)]
    if mode=='random':
        return [secrets.randbits(44) for _ in range(L)]
    raise ValueError(mode)


def fixtures():
    yield 'positive_endpoint', [(Q,50303,('random','random'))]
    yield 'negative_endpoint_overflow', [(-Q,0,('overflow','accept'))]
    yield 'both_factories_and_first_success', [
        (0,7,('accept','reject')), (0,7,('reject','accept')),
        (-Q,9,('accept','accept')), (Q,10,('random','random'))]
    for name in ('early','late','all_fail','joint_uniform'):
        rows=[]
        for i in range(201):
            if name=='early':
                rows.append((Q if i==0 else 0, (37*i)%50304, ('random','random') if i==0 else ('reject','reject')))
            elif name=='late':
                rows.append((-Q if i==200 else 0, (37*i)%50304, ('accept','accept') if i==200 else ('reject','accept')))
            elif name=='all_fail':
                rows.append((-Q,(37*i)%50304,('overflow','accept')))
            else:
                rows.append((secrets.randbelow(2*Q+1)-Q,secrets.randbelow(50304),('random','random')))
        yield 'full201_'+name,rows


def main():
    registration=ROOT/'notes/20261030_bounded_accept_select_preregistration_v2.json'
    spec=json.loads(registration.read_bytes())
    for name,digest in spec['input_sha256'].items():
        assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/bounded_accept_select.json'
    assert not destination.exists()
    folder=ROOT/'results/bounded_accept_select_runs';folder.mkdir()
    binary=ROOT/'toolchain/bounded-sampler/bounded_accept_select'
    rows=[]
    for name,case in fixtures():
        sub=folder/name;sub.mkdir()
        data=[bytearray(),bytearray()]
        indices=[];accepted=[];overflows=0
        for delta,index,modes in case:
            coins=[make_chain(delta,mode) for mode in modes]
            bit,overflow=acceptance(delta,coins)
            indices.append(index);accepted.append(bit);overflows+=overflow
            words=[(delta&((1<<22)-1),22,4),(index,16,4)]
            words += [(u,44,8) for chain in coins for u in chain]
            for value,width,size in words:
                mask=secrets.randbits(width)
                data[0].extend(mask.to_bytes(size,'little'))
                data[1].extend((value^mask).to_bytes(size,'little'))
        expected=first_success(indices,accepted,65536)
        inputs=[sub/f'party{i+1}_input.bin' for i in range(2)]
        outputs=[sub/f'party{i+1}_output.txt' for i in range(2)]
        metrics=[sub/f'party{i+1}_metrics.json' for i in range(2)]
        for path,value in zip(inputs,data):path.write_bytes(value)
        port=transport.port()
        commands=[[str(binary),str(i+1),str(port),str(len(case)),str(inputs[i]),str(outputs[i]),str(metrics[i])]
                  for i in range(2)]
        record=transport.pair(commands,sub,timeout=180)
        assert outputs[0].read_bytes()==b''
        actual=int(outputs[1].read_text())
        assert actual==expected,(name,actual,expected)
        party=[json.loads(p.read_bytes()) for p in metrics]
        assert party[0]['sent']==party[1]['received'] and party[1]['sent']==party[0]['received']
        assert party[0]['and_gates']==party[1]['and_gates']
        record.update(name=name,count=len(case),all_correct=True,expected_final_token=expected,
            controlled_overflow_proposals=overflows,owner_output_empty=True,
            total_wire_bytes=sum(x['sent'] for x in party),
            max_party_protocol_ms=max(x['protocol_ms'] for x in party),parties=party,
            input_share_sha256=[sha(p) for p in inputs],output_sha256=[sha(p) for p in outputs])
        rows.append(record)
        print(name,'correct',actual,'bytes',record['total_wire_bytes'],'ms',record['max_party_protocol_ms'],flush=True)
    full=[r for r in rows if r['count']==201]
    assert len({r['parties'][0]['and_gates'] for r in full})==1
    assert len({r['total_wire_bytes'] for r in full})==1
    out=dict(all_passed=True,results=rows,full201_fixed_traffic_and_gate_count=True,
        preregistration_sha256=sha(registration),input_sha256=spec['input_sha256'],limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,actual_pairs=len(rows),full201_pairs=len(full)),indent=2),flush=True)


if __name__=='__main__':main()
