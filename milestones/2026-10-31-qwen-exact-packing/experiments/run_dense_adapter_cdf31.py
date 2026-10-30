"""Complete dense private-head sampler for the supplementary matched E31 law."""
import json
from pathlib import Path

import numpy as np
from bounded_adapter_fixed16 import encode_parameters,tanh_integer,SCALE
from finite_alias_table import build
from run_dense_adapter_linearot import fixtures as scoring_fixtures,input_bytes,port_triple,reconstruct,sha
from run_private_weighted_cdf93_check import audit as audit_cdf
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def exp31(delta,tables):
    y=(1<<20)-delta
    t=[np.array(a,dtype=np.int64) for a in tables]
    values=(t[0][y&255]*t[1][(y>>8)&255])>>31
    return (values*t[2][(y>>16)&63])>>31


def main():
    reg=ROOT/'notes/20261030_dense_adapter_cdf31_preregistration.json';spec=json.loads(reg.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    cert=json.loads((ROOT/'results/sci_exp31_finite_certificate.json').read_bytes());assert cert['all_passed']
    controls=json.loads((ROOT/'results/private_weighted_cdf93_check.json').read_bytes());assert controls['all_passed']
    destination=ROOT/'results/dense_adapter_cdf31.json';assert not destination.exists()
    folder=ROOT/'results/dense_adapter_cdf31_runs';folder.mkdir();records=[]
    with np.load(ROOT/'results/bounded_adapter_gc_inputs.npz') as archive:qs=archive['q']
    for name,x,b,a in scoring_fixtures():
        n=len(a)
        if name.startswith('ptb') or name.startswith('wiki'):
            q=qs[0 if name.startswith('ptb') else 1]
            cut,alternate,mass,represented,diagnostics=build(q,44)
            qmass=mass[:50304].astype(np.uint64)
        else:
            qmass=np.full(n,(1<<60)//n,dtype=np.uint64)
            qmass[:(1<<60)%n]+=1;diagnostics=None
        assert sum(map(int,qmass))==1<<60
        raw_t=x@b.T;t=raw_t//SCALE;raw_z=a@t;delta=tanh_integer(raw_z//SCALE)
        expected_exp=exp31(delta,cert['actual_public_factors']['tables'])
        sub=folder/name;sub.mkdir();paths=[sub/f'party{i}' for i in (1,2)]
        for p in paths:p.mkdir()
        for role,p in enumerate(paths,1):
            data=input_bytes(x,b,a,role)
            qm=qmass if role==2 else np.zeros(n,dtype=np.uint64)
            (p/'own_input.bin').write_bytes(data+qm.astype('<u8').tobytes())
        port=port_triple();binary=ROOT/'toolchain/bounded-sampler/dense_adapter_cdf31'
        commands=[[str(binary),str(role),str(port),str(n),str(paths[role-1]/'own_input.bin'),str(paths[role-1]),'1'] for role in (1,2)]
        row=transport.pair(commands,sub,timeout=600)
        for filename,width,expected in [('raw_t_add41.bin',41,raw_t),('t_add26.bin',26,t),
            ('raw_z_add44.bin',44,raw_z),('delta_add44.bin',44,delta)]:
            assert np.array_equal(reconstruct(paths,filename,width),expected),(name,filename)
        # exp31 is unsigned33, below2^31 inclusive, so signed decode is safe.
        assert np.array_equal(reconstruct(paths,'exp_add33.bin',33),expected_exp),(name,'exp31')
        row.update(audit_cdf(sub,qmass.tolist(),expected_exp.tolist()))
        party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        for channel in range(3):
            assert party[0]['end_counters'][channel]==party[1]['end_counters'][channel+3]
            assert party[1]['end_counters'][channel]==party[0]['end_counters'][channel+3]
        stages={};previous=0
        for key,stage in zip(['setup_counters','bx_counters','bridge_counters','az_counters','end_counters'],
            ['setup','Bx','floor_bridge','A_t','PWL_exp_weighted_CDF_release']):
            value=sum(sum(p[key][:3]) for p in party);stages[stage]=value-previous;previous=value
        row.update(name=name,all_shared_scores_exact=True,all_exp31_values_exact=True,parties=party,
            total_wire_bytes=previous,stage_wire_bytes=stages,
            max_party_protocol_ms=max(p['protocol_ms'] for p in party),finite_alias=diagnostics,
            files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(row)
        print(name,'complete private denseCDF correct, bytes',previous,'ms',row['max_party_protocol_ms'],flush=True)
    out=dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],preregistration_sha256=sha(reg),limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_passed=True,complete_pairs=len(records))))


if __name__=='__main__':main()
