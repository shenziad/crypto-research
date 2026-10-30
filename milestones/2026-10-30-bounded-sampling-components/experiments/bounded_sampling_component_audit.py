"""Read-only audit of completed components; no full-pipeline claim."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import numpy as np

from finite_descending_bernoulli import acceptance,first_success
from bounded_adapter_fixed16 import tanh_integer
from run_bounded_adapter_gc import fixtures

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(v,width):return v-(1<<width) if v & (1<<(width-1)) else v


def main():
    destination=ROOT/'results/bounded_sampling_component_audit.json';assert not destination.exists()
    checked_hashes=0
    for stem in ('bounded_adapter_fixed16_quality','fable_smallbatch_balanced','bounded_accept_select','bounded_adapter_gc'):
        result=json.loads((ROOT/'results'/f'{stem}.json').read_bytes())
        for name,digest in result['input_sha256'].items():
            assert sha(ROOT/name)==digest,name
            checked_hashes+=1
    cert=json.loads((ROOT/'results/bounded_adapter_fixed16_certificate.json').read_bytes())
    ranges=cert['integer_range_bounds']
    d=json.loads((ROOT/'results/bounded_adapter_gc.json').read_bytes())
    rows={r['name']:r for r in d['results']}
    exact_bigint_intermediates=0;correct_token_only_runs=0;input_records=0
    all_rows=[]
    for name,mode,x,b,a,indices in fixtures():
        sub=ROOT/'results/bounded_adapter_gc_runs'/name
        data=[(sub/f'party{i+1}_input.bin').read_bytes() for i in range(2)]
        positions=[0,0]
        def reconstruct(width,size):
            values=[]
            for i in range(2):
                at=positions[i];values.append(int.from_bytes(data[i][at:at+size],'little'))
                positions[i]+=size
            return (values[0]^values[1])&((1<<width)-1)
        xx=[signed(reconstruct(22,4),22) for _ in range(128)]
        bb=[[signed(reconstruct(18,4),18) for _ in range(128)] for _ in range(8)]
        assert xx==x.tolist() and bb==b.tolist()
        # All concrete fixtures actually satisfy the stricter ||xi/S||2<=20.
        assert sum(v*v for v in xx) <= (20*65536)**2
        assert max(map(abs,xx))<=ranges['max_abs_input_integer']
        raw_t=[sum(u*v for u,v in zip(xx,row)) for row in bb]
        t=[v//65536 for v in raw_t]
        assert max(map(abs,raw_t))<=ranges['max_abs_t_accumulator']
        assert max(map(abs,t))<=ranges['max_abs_t_integer']
        zs=[];deltas=[];accepted=[];ii=[]
        for expected_a,expected_i in zip(a,indices):
            aa=[signed(reconstruct(18,4),18) for _ in range(8)]
            index=reconstruct(16,4)
            coins=[[reconstruct(44,8) for _ in range(19)] for _ in range(2)]
            assert aa==expected_a.tolist() and index==expected_i
            raw=sum(u*v for u,v in zip(aa,t));z=raw//65536
            assert abs(raw)<=ranges['max_abs_z_accumulator'] and abs(z)<=ranges['max_abs_z_integer']
            delta=int(tanh_integer(np.array([z],dtype=np.int64))[0])
            zs.append(z);deltas.append(delta);ii.append(index)
            accepted.append(acceptance(delta,coins)[0]);input_records+=1
        assert positions==list(map(len,data))
        if mode=='shares':
            values=[np.frombuffer((sub/f'party{i+1}_output.bin').read_bytes(),dtype='<u4') for i in range(2)]
            actual=(values[0]^values[1]).view('<i4').tolist()
            assert actual==t+zs+deltas
            exact_bigint_intermediates+=len(actual)
        else:
            assert (sub/'party1_output.bin').read_bytes()==b''
            assert int((sub/'party2_output.bin').read_text())==first_success(ii,accepted,65536)
            correct_token_only_runs+=1
        for i in range(2):
            assert sha(sub/f'party{i+1}_input.bin')==rows[name]['input_share_sha256'][i]
            assert sha(sub/f'party{i+1}_output.bin')==rows[name]['output_sha256'][i]
        all_rows.append(dict(name=name,quantized_input_l2=float(sum(v*v for v in xx)**.5/65536),
            max_abs_t_accumulator=max(map(abs,raw_t)),range_certificate_passed=True))
    budget=json.loads((ROOT/'results/finite_descending_bernoulli_check.json').read_bytes())['sampling_error_budget']
    total=sum((F(budget[n]['numerator'],budget[n]['denominator']) for n in
               ('no_success','factory_truncation_coupling','temperature_tv')),F(0))
    assert total==F(budget['total']['numerator'],budget['total']['denominator'])<F(1,1<<40)
    out=dict(all_passed=True,input_file_hashes_checked=checked_hashes,
        private_input_records_parsed=input_records,exact_bigint_intermediates=exact_bigint_intermediates,
        correct_token_only_runs=correct_token_only_runs,concrete_domain_checks=all_rows,
        exact_sampling_error_budget=float(total),
        source_sha256=sha(Path(__file__)),
        limits='Audit validates component records and exact private arithmetic references only. Does not establish fresh quality evidence, full protocol composition, implementation security, end-to-end speed or independent novelty.')
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
