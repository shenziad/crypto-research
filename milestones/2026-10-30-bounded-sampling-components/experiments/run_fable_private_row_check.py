"""Pinned FABLE smoke tests and exact private wide-row interface checks.

No proposal sampling, private adapter arithmetic or end-to-end speed claim.
The parent supplies test inputs and reconstructs only after both parties exit.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'third_party/FABLE'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def pair(commands, folder, timeout=600):
    paths=[folder/f'party{i+1}.log' for i in range(2)]
    files=[p.open('xb') for p in paths]
    processes=[]
    started=time.monotonic()
    try:
        for cmd,f in zip(commands,files):
            processes.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=SOURCE))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('one party exited with failure; see retained logs')
            if time.monotonic()-started>timeout:
                raise TimeoutError('registered pair timeout')
            time.sleep(.25)
        assert all(p.returncode==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:p.wait(5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in files:f.close()
    return dict(wall_seconds=time.monotonic()-started,
                logs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])


def main():
    prereg_path=ROOT/'notes/20261030_fable_private_row_preregistration_v3.json'
    prereg=json.loads(prereg_path.read_bytes())
    for name,h in prereg['source_sha256'].items():
        assert sha(ROOT/name)==h,name
    for name,h in prereg['binary_sha256'].items():
        assert sha(SOURCE/'build/bin'/name)==h,name
    destination=ROOT/'results/fable_private_row_check.json'
    assert not destination.exists()
    folder=ROOT/'results/fable_private_row_check_runs';assert not folder.exists()
    folder.mkdir()
    os.environ['OMP_NUM_THREADS']='4'
    smoke=[]
    for name,args,markers in [('aes',['s=16'],['Test Passed!']),
                              ('deduplication',['s=256'],['[Deduplication] Test passed'])]:
        sub=folder/name;sub.mkdir();p=port()-100
        commands=[[str(SOURCE/'build/bin'/name),f'r={role}',f'p={p}']+args for role in (1,2)]
        row=pair(commands,sub)
        for entry in row['logs']:
            text=(ROOT/entry['path']).read_text(errors='replace')
            assert all(marker in text for marker in markers),(name,text[-2000:])
        row['name']=name;row['all_passed']=True;smoke.append(row)
        print('upstream smoke passed',name,flush=True)
    n=65536;batch=256;width=32
    # Independent full-byte control table; not an LLM distribution experiment.
    control=np.frombuffer(b''.join(hashlib.sha256(b'FABLE wide-row control\0'+i.to_bytes(4,'little')).digest()
                                  for i in range(n)),dtype=np.uint8).reshape(n,width).copy()
    archive=ROOT/'results/bounded_output_adapter_weights.npz'
    assert sha(archive)==prereg['adapter_weights_sha256']
    with np.load(archive) as z:
        a=np.ascontiguousarray(z['bounded_A'],dtype='<f4')
    assert a.shape==(50304,8)
    actual=np.zeros((n,width),dtype=np.uint8)
    actual[:50304]=np.frombuffer(a.tobytes(),dtype=np.uint8).reshape(50304,width)
    cases=[]
    for label,table in [('synthetic_wide',control),('frozen_A_191_plus65',actual)]:
        sub=folder/label;sub.mkdir()
        if label=='synthetic_wide':
            indices=[0,65535,32768,1]+[secrets.randbelow(n) for _ in range(batch-4)]
            indices[128:]=indices[:128]  # prescribed duplicate control
        else:
            pool=[secrets.randbelow(50304) for _ in range(64)]
            indices=[0,50303,50304,65535]+[secrets.choice(pool) for _ in range(187)]+[0]*65
        indices=np.asarray(indices,dtype='<u2');assert len(indices)==batch
        owner=np.frombuffer(os.urandom(2*batch),dtype='<u2').copy()
        other=np.bitwise_xor(owner,indices)
        tablefile=sub/'owner_table.bin';tablefile.write_bytes(table.tobytes())
        ownerfile=sub/'owner_index_shares.bin';ownerfile.write_bytes(owner.tobytes())
        otherfile=sub/'other_index_shares.bin';otherfile.write_bytes(other.tobytes())
        p=port()
        out=[sub/f'party{i+1}_row_shares.bin' for i in range(2)]
        metrics=[sub/f'party{i+1}_metrics.json' for i in range(2)]
        commands=[[str(SOURCE/'build/bin/private_row_harness'),str(i+1),str(p),str(batch),
                   str(tablefile) if i==0 else '-',str(ownerfile if i==0 else otherfile),
                   str(out[i]),str(metrics[i]),'4'] for i in range(2)]
        row=pair(commands,sub)
        shares=[np.frombuffer(f.read_bytes(),dtype=np.uint8).reshape(batch,width) for f in out]
        reconstructed=np.bitwise_xor(*shares)
        assert np.array_equal(reconstructed,table[indices])
        party=[json.loads(f.read_bytes()) for f in metrics]
        assert party[0]['sent']==party[1]['received'] and party[1]['sent']==party[0]['received']
        row.update(name=label,queries=batch,effective_sampler_queries=191 if label!='synthetic_wide' else None,
                   all_bytes_exact=True,all_correct=True,parties=party,
                   total_wire_bytes=sum(x['sent'] for x in party),
                   max_party_protocol_ms=max(x['protocol_ms'] for x in party),
                   table_sha256=sha(tablefile),input_share_sha256=[sha(ownerfile),sha(otherfile)],
                   output_share_sha256=[sha(f) for f in out])
        cases.append(row)
        print(label,'all256 wide rows exact, bytes',row['total_wire_bytes'],flush=True)
    result=dict(all_passed=True,upstream_smoke=smoke,results=cases,
                preregistration_sha256=sha(prereg_path),source_sha256=prereg['source_sha256'],
                binary_sha256=prereg['binary_sha256'],adapter_weights_sha256=sha(archive),
                limits=prereg['limits'])
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_passed=True,fullsize_queries_verified=512),indent=2))


if __name__=='__main__':main()
