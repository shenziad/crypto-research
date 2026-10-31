"""Three registered realtwo-party controls, always audit after both exits."""
import json
from pathlib import Path
import secrets

from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
import numpy as np

from fable_proof_integration_budget import reserve,finish
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def run_pair(label,commands,folder):
    remaining=reserve(label,1)
    try:
        record=transport.pair(commands,folder,timeout=min(900,remaining))
    except BaseException:
        finish(label,False);raise
    return record


def main():
    reg=ROOT/'notes/20261031_fable_proof_integration_preregistration.json'
    spec=json.loads(reg.read_bytes())
    for name,h in spec['input_sha256'].items():assert transport.sha(ROOT/name)==h,name
    dest=ROOT/'results/fable_proof_controls.json';assert not dest.exists()
    base=ROOT/'results/fable_proof_control_runs';base.mkdir();records=[]
    folder=base/'AES';folder.mkdir();paths=[folder/f'party{i}' for i in (1,2)]
    for p in paths:p.mkdir()
    ids=np.array([0,1,2,65535,65536,151935,262143,262145,262399,131072,262142,262146,42,2,0,262399],dtype='<u4')
    key=secrets.token_bytes(16);masks=np.array([secrets.randbits(19) for _ in ids],dtype='<u4')
    for i,p in enumerate(paths):(p/'own.bin').write_bytes((key if i==0 else bytes(16))+(masks if i==0 else masks^ids).astype('<u4').tobytes())
    port=transport.port();binary=ROOT/'third_party/FABLE-qwen-proof-64/build/bin/dual_domain_aes_control'
    commands=[[str(binary),str(i+1),str(port),str(p/'own.bin'),str(p/'tags.bin'),str(p/'metrics.json'),str(len(ids))] for i,p in enumerate(paths)]
    label='control:dual_domain_AES';record=run_pair(label,commands,folder)
    enc=Cipher(algorithms.AES(key),modes.ECB()).encryptor()
    plain=b''.join((int(i)+(d<<127)).to_bytes(16,'little') for i in ids for d in (0,1))
    expected=enc.update(plain)+enc.finalize()
    assert (paths[0]/'tags.bin').read_bytes()==b'' and (paths[1]/'tags.bin').read_bytes()==expected
    metrics=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
    assert metrics[0]['sent']==metrics[1]['received'] and metrics[1]['sent']==metrics[0]['received']
    finish(label,True);record.update(name='AES_domains',all_correct=True,IDs_checked=len(ids),domains=2,total_wire_bytes=sum(m['sent'] for m in metrics),parties=metrics)
    records.append(record);print('ActualGCdual-domainAES agreeswithindependentAES on32blocks',flush=True)
    for width in (64,144):
        folder=base/f'LUT{width}';folder.mkdir();paths=[folder/f'party{i}' for i in (1,2)]
        for p in paths:p.mkdir()
        rng=np.random.default_rng(20261031+width)
        table=rng.integers(0,256,size=(262144,width//8),dtype=np.uint8)
        indices=rng.integers(0,262144,size=256,dtype=np.uint32)
        indices[:128]=262143;indices[128:224]=0
        indices[224:232]=[0,65535,65536,131072,151935,262142,1,2]
        # Highest-IDduplicates reach final sortedposition255 and virtualID262399.
        sorted_ids=np.sort(indices);virtual=np.where(np.r_[False,sorted_ids[1:]==sorted_ids[:-1]],262144+np.arange(256),sorted_ids)
        assert len(set(map(int,virtual)))==256 and int(virtual.max())==262399 and 262144 not in virtual
        masks=np.array([secrets.randbits(18) for _ in indices],dtype='<u4')
        (paths[0]/'table.bin').write_bytes(table.tobytes())
        for i,p in enumerate(paths):(p/'indices.bin').write_bytes((masks if i==0 else masks^indices).astype('<u4').tobytes())
        binary=ROOT/f'third_party/FABLE-qwen-proof-{width}/build/bin/private_row_harness';port=transport.port()
        commands=[[str(binary),str(i+1),str(port),'256',str(p/'table.bin') if i==0 else '-',str(p/'indices.bin'),str(p/'rows.bin'),str(p/'metrics.json'),'4'] for i,p in enumerate(paths)]
        label=f'control:LUT{width}';record=run_pair(label,commands,folder)
        rows=[np.frombuffer((p/'rows.bin').read_bytes(),dtype=np.uint8).reshape(256,width//8) for p in paths]
        assert np.array_equal(rows[0]^rows[1],table[indices])
        metrics=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert all(m['proof_candidate'] and m['covered_rows']==262399 and m['position_slots']==4096 and m['flood_bits']==149 and m['fresh_publickey_zero'] for m in metrics)
        assert metrics[0]['sent']==metrics[1]['received'] and metrics[1]['sent']==metrics[0]['received']
        finish(label,True);record.update(name=f'LUT{width}',all_correct=True,rows_checked=256,virtual_ID_max_checked=262399,total_wire_bytes=sum(m['sent'] for m in metrics),parties=metrics)
        records.append(record);print(f'PrivateLUT{width}: all256rows exact; maximumvirtualIDcovered; bytes={record["total_wire_bytes"]}',flush=True)
    files={str(p.relative_to(ROOT)):transport.sha(p) for p in base.rglob('*') if p.is_file()}
    dest.write_text(json.dumps(dict(all_passed=True,results=records,native_pairs=3,files=files,preregistration_sha256=transport.sha(reg),limits=spec['limits']),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        ledger=json.loads((ROOT/'results/fable_proof_integration_ledger.json').read_bytes())
        for event in ledger['events']:
            if 'completed_utc' not in event:finish(event['label'],False)
        raise
