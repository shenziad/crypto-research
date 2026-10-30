"""Actual private weighted-CDF controls and reusable after-exit audit."""
import json
import os
from pathlib import Path
import secrets

import numpy as np
from run_dense_adapter_linearot import port_triple,sha
import run_fable_smallbatch_check as transport
from qwen_stage_budget import reserve

ROOT=Path(__file__).resolve().parents[1]


def read_words(path,size):
    data=path.read_bytes();assert len(data)%size==0
    return [int.from_bytes(data[i:i+size],'little') for i in range(0,len(data),size)]


def audit(folder,q,e):
    paths=[folder/f'party{i}' for i in (1,2)];n=len(q)
    for name,width,expected in [('weight_low_add63.bin',63,[int(a)*int(b&((1<<30)-1)) for a,b in zip(e,q)]),
        ('weight_mid_add63.bin',63,[int(a)*int((b>>30)&((1<<30)-1)) for a,b in zip(e,q)]),
        ('weight_high_add36.bin',36,[int(a)*int(b>>60) for a,b in zip(e,q)])]:
        a,b=[read_words(p/name,8) for p in paths]
        assert [((x+y)&((1<<width)-1)) for x,y in zip(a,b)]==expected,name
    weights=[int(a)*int(b) for a,b in zip(q,e)];prefix=[];total=0
    for value in weights:total+=value;prefix.append(total)
    assert 0<total<=1<<93
    a,b=[read_words(p/'prefix_add95.bin',16) for p in paths]
    assert [((x+y)&((1<<95)-1)) for x,y in zip(a,b)]==prefix
    coin=[read_words(p/'uniform_share.bin',8)[0] for p in paths];uniform=coin[0]^coin[1]
    threshold=(total*uniform)>>64
    shares=[read_words(p/'r_add95.bin',16)[0] for p in paths]
    assert (sum(shares)&((1<<95)-1))==threshold
    flags=[(p/'cdf_negative_xor.bin').read_bytes() for p in paths]
    assert len(flags[0])==len(flags[1])==n
    assert [x^y for x,y in zip(*flags)]==[int(threshold<p) for p in prefix]
    expected=next(i for i,p in enumerate(prefix) if p>threshold)
    assert (paths[0]/'final_output.txt').read_bytes()==b''
    assert int((paths[1]/'final_output.txt').read_text())==expected
    return dict(all_correct=True,rows=n,prefix_values_checked=n,products_checked=2*n,
        final_token_matches=True,normalizer_bits=total.bit_length())


def fixtures():
    s=1<<31;n=16
    for name,index,u in [('peak0_u0',0,0),('peaklast_umax',15,(1<<64)-1)]:
        length=151936 if index==15 else n
        q=[0]*length;q[-1 if index==15 else 0]=1<<62;yield name,q,[s]*length,u
    yield 'uniform_half',[1<<58]*16,[s]*16,1<<63
    q=[0]*16;q[2]=q[7]=1<<61
    yield 'zero_support_interior',q,[s]*16,1<<63
    q=[0]*16;q[0]=1;q[-1]=(1<<62)-1
    yield 'one_mass_at_zero',q,[s//2+i for i in range(16)],0
    rng=np.random.default_rng(20261030);mass=rng.integers(1,1<<20,size=128).tolist()
    cumulative=[0];total=sum(mass);running=0
    for value in mass:running+=value;cumulative.append((running*(1<<62))//total)
    q=[b-a for a,b in zip(cumulative[:-1],cumulative[1:])]
    yield 'random_q_exp',q,rng.integers(1<<29,(1<<31)+1,size=128).tolist(),secrets.randbits(64)


def main():
    reg=ROOT/'notes/20261030_private_weighted_cdf95_preregistration.json';spec=json.loads(reg.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/private_weighted_cdf95_check.json';assert not destination.exists()
    folder=ROOT/'results/private_weighted_cdf95_check_runs';folder.mkdir();records=[]
    for name,q,e,uniform in fixtures():
        sub=folder/name;sub.mkdir();paths=[sub/f'party{i}' for i in (1,2)]
        masks=[secrets.randbits(33) for _ in e];coins=[secrets.randbits(64),0];coins[1]=coins[0]^uniform
        for j,p in enumerate(paths):
            p.mkdir();data=bytearray()
            for i,value in enumerate(e):
                share=masks[i] if j==0 else (value-masks[i])&((1<<33)-1)
                data.extend(share.to_bytes(8,'little'));data.extend((0 if j==0 else q[i]).to_bytes(8,'little'))
            data.extend(coins[j].to_bytes(8,'little'));(p/'own_input.bin').write_bytes(data)
        port=port_triple();binary=ROOT/'toolchain/bounded-sampler/private_weighted_cdf95_check'
        commands=[[str(binary),str(j+1),str(port),str(len(q)),str(p/'own_input.bin'),str(p)] for j,p in enumerate(paths)]
        remaining=reserve("CDF95:"+name,1)
        row=transport.pair(commands,sub,timeout=min(180,remaining));row.update(audit(sub,q,e))
        parties=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert parties[0]['sent']==parties[1]['received'] and parties[1]['sent']==parties[0]['received']
        row.update(name=name,parties=parties,total_wire_bytes=sum(p['sent'] for p in parties),
            files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(row);print(name,'privateCDF correct, bytes',row['total_wire_bytes'],flush=True)
    out=dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],preregistration_sha256=sha(reg),limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_passed=True,pairs=len(records))))


if __name__=='__main__':main()
