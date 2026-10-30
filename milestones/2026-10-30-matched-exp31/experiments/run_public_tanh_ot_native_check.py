import json
import os
from pathlib import Path

import numpy as np
from bounded_adapter_fixed16 import tanh_integer
from run_dense_adapter_linearot import port_triple,reconstruct,sha
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]

def main():
    reg=ROOT/'notes/20261030_public_tanh_ot_native_preregistration.json';spec=json.loads(reg.read_bytes())
    for name,digest in spec['input_sha256'].items():assert sha(ROOT/name)==digest,name
    destination=ROOT/'results/public_tanh_ot_native_check.json';assert not destination.exists()
    z=np.concatenate((np.arange(-4*65536-1,4*65536+2,dtype=np.int64),np.array([-118636417,118636417,-((1<<27)-1),(1<<27)-1],dtype=np.int64)))
    # Alternate exact and just-below-next-integer inputs to exercise signed floor.
    raw=(z*65536+((np.arange(len(z))&1)*65535))&((1<<44)-1)
    first=np.frombuffer(os.urandom(8*len(z)),dtype='<u8').copy()&((1<<44)-1)
    second=(raw.astype(np.uint64)-first)&((1<<44)-1)
    folder=ROOT/'results/public_tanh_ot_native_check_runs';folder.mkdir()
    paths=[folder/f'party{i}' for i in (1,2)]
    for p,v in zip(paths,(first,second)):p.mkdir();(p/'own_input.bin').write_bytes(v.astype('<u8').tobytes())
    port=port_triple();binary=ROOT/'toolchain/bounded-sampler/public_tanh_ot_native_check'
    commands=[[str(binary),str(i+1),str(port),str(p/'own_input.bin'),str(p/'delta_add44.bin'),str(p/'metrics.json')] for i,p in enumerate(paths)]
    row=transport.pair(commands,folder,timeout=600)
    assert np.array_equal(reconstruct(paths,'delta_add44.bin',44),tanh_integer(z))
    party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
    assert party[0]['sent']==party[1]['received'] and party[1]['sent']==party[0]['received']
    out=dict(all_passed=True,actual_two_party_values=len(z),total_wire_bytes=sum(p['sent'] for p in party),parties=party,
        execution=row,preregistration_sha256=sha(reg),input_sha256=spec['input_sha256'],
        files={str(p.relative_to(ROOT)):sha(p) for folder in paths for p in folder.iterdir() if p.is_file()},limits=spec['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_passed=True,values=len(z),bytes=out['total_wire_bytes'])))

if __name__=='__main__':main()
