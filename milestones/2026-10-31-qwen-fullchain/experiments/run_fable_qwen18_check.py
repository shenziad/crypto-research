"""Two actual 18-bit private-row tests of the isolated pinned FABLE variant."""
from datetime import datetime,timezone
import json
from pathlib import Path
import secrets

import numpy as np
from qwen_stage_budget import reserve
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def main():
    reg=ROOT/'notes/20261030_fable_qwen18_preregistration.json';spec=json.loads(reg.read_bytes())
    for n,h in spec['input_sha256'].items():assert transport.sha(ROOT/n)==h,n
    destination=ROOT/'results/fable_qwen18_check.json';assert not destination.exists()
    folder=ROOT/'results/fable_qwen18_check_runs';folder.mkdir();records=[]
    binary=ROOT/'third_party/FABLE-qwen18/build/bin/private_row_harness'
    for j,name in enumerate(('fullrange_high_indices','duplicates_and_padding')):
        sub=folder/name;sub.mkdir();paths=[sub/f'party{i}' for i in (1,2)]
        for p in paths:p.mkdir()
        rng=np.random.default_rng(20261031+j)
        table=rng.integers(0,256,size=(262144,32),dtype=np.uint8)
        indices=rng.integers(0,262144,size=256,dtype=np.uint32)
        if j==0:indices[:6]=[0,65535,65536,131072,151935,262143]
        else:
            indices[:201]=np.resize(np.array([0,151935,65536,151935,262143],dtype=np.uint32),201)
            indices[201:]=0
        masks=np.array([secrets.randbits(18) for _ in range(256)],dtype='<u4')
        (paths[0]/'owner_table.bin').write_bytes(table.tobytes())
        (paths[0]/'index_shares.bin').write_bytes(masks.tobytes())
        (paths[1]/'index_shares.bin').write_bytes((masks^indices).astype('<u4').tobytes())
        port=transport.port()
        commands=[[str(binary),str(i+1),str(port),'256',str(p/'owner_table.bin') if i==0 else '-',
            str(p/'index_shares.bin'),str(p/'row_shares.bin'),str(p/'metrics.json'),'4'] for i,p in enumerate(paths)]
        remaining=reserve('FABLE18:'+name,1)
        row=transport.pair(commands,sub,timeout=min(600,remaining))
        # No files are inspected until both native participants return.
        shares=[np.frombuffer((p/'row_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,32) for p in paths]
        assert np.array_equal(shares[0]^shares[1],table[indices])
        metrics=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert metrics[0]['sent']==metrics[1]['received'] and metrics[1]['sent']==metrics[0]['received']
        row.update(name=name,all_rows_correct=True,rows_checked=256,parties=metrics,
            total_wire_bytes=sum(p['sent'] for p in metrics),
            files={str(p.relative_to(ROOT)):transport.sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(row);print(name,'FABLE18 rows correct, bytes',row['total_wire_bytes'],flush=True)
    out=dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],
        preregistration_sha256=transport.sha(reg),limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
