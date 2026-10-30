"""Two packed private-LUT pairs and one alias-boundary pair, after-exit audit."""
import json
from pathlib import Path
import secrets

import numpy as np
from qwen_exact_row_codec import pack18,unpack18
from qwen_exactpacking_budget import reserve
import run_fable_smallbatch_check as transport

ROOT=Path(__file__).resolve().parents[1]


def main():
    reg=ROOT/'notes/20261031_qwen_exactpacking_controls_preregistration.json'
    spec=json.loads(reg.read_bytes())
    for n,h in spec['input_sha256'].items():assert transport.sha(ROOT/n)==h,n
    destination=ROOT/'results/qwen_exactpacking_controls.json';assert not destination.exists()
    base=ROOT/'results/qwen_exactpacking_controls_runs';base.mkdir();records=[]
    for width in (64,144):
        name=f'LUT{width}';folder=base/name;folder.mkdir();paths=[folder/f'party{i}' for i in (1,2)]
        for p in paths:p.mkdir()
        rng=np.random.default_rng(20261031+width)
        table=rng.integers(0,256,size=(262144,width//8),dtype=np.uint8)
        indices=rng.integers(0,262144,size=256,dtype=np.uint32)
        indices[:8]=[0,65535,65536,131072,151935,262143,151935,0]
        indices[201:]=0
        masks=np.array([secrets.randbits(18) for _ in indices],dtype='<u4')
        (paths[0]/'table.bin').write_bytes(table.tobytes())
        for j,p in enumerate(paths):
            (p/'index_shares.bin').write_bytes((masks if j==0 else masks^indices).astype('<u4').tobytes())
        port=transport.port();binary=ROOT/f'third_party/FABLE-qwen18-{width}/build/bin/private_row_harness'
        commands=[[str(binary),str(j+1),str(port),'256',str(p/'table.bin') if j==0 else '-',
            str(p/'index_shares.bin'),str(p/'row_shares.bin'),str(p/'metrics.json'),'4'] for j,p in enumerate(paths)]
        remaining=reserve('control:'+name,1);row=transport.pair(commands,folder,timeout=min(600,remaining))
        shares=[np.frombuffer((p/'row_shares.bin').read_bytes(),dtype=np.uint8).reshape(256,width//8) for p in paths]
        assert np.array_equal(shares[0]^shares[1],table[indices])
        if width==144:
            assert np.array_equal(unpack18(shares[0])^unpack18(shares[1]),unpack18(table[indices]))
            assert np.array_equal(pack18(unpack18(table[indices])),table[indices])
        parties=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
        assert all(p['row_bytes']==width//8 for p in parties)
        assert parties[0]['sent']==parties[1]['received'] and parties[1]['sent']==parties[0]['received']
        row.update(name=name,all_correct=True,rows_checked=256,parties=parties,
            total_wire_bytes=sum(p['sent'] for p in parties),
            files={str(p.relative_to(ROOT)):transport.sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
        records.append(row);print(name,'all256 private rows exact, bytes',row['total_wire_bytes'],flush=True)

    folder=base/'alias_boundaries';folder.mkdir();paths=[folder/f'party{i}' for i in (1,2)]
    data=[bytearray(),bytearray()];expected=[];cuts=[0,1,(1<<44)-1,1<<44]
    for i in range(201):
        cut=cuts[(i//4)%4];mode=i%4
        uniform=[0,max(0,cut-1),min(cut,(1<<44)-1),(1<<44)-1][mode]
        column=(65536+257*i)&((1<<18)-1);alternate=(151935-101*i)&((1<<18)-1)
        expected.append(column if uniform<cut else alternate)
        row=cut|(alternate<<45)
        cm=secrets.randbits(18);rm=secrets.randbits(64);um=secrets.randbits(44);pad=secrets.randbits(18)
        for j in range(2):
            data[j].extend((cm if j==0 else cm^column).to_bytes(4,'little'))
            data[j].extend((rm if j==0 else rm^row).to_bytes(8,'little')+bytes(24))
            data[j].extend((um if j==0 else um^uniform).to_bytes(8,'little'))
            data[j].extend((pad if j==0 else 0).to_bytes(4,'little'))
    for p,value in zip(paths,data):p.mkdir();(p/'own_input.bin').write_bytes(value)
    binary=ROOT/'toolchain/bounded-sampler/qwen_alias_batch';port=transport.port()
    commands=[[str(binary),str(j+1),str(port),str(p/'own_input.bin'),str(p/'index_shares.bin'),str(p/'metrics.json')] for j,p in enumerate(paths)]
    remaining=reserve('control:alias_boundaries',1);row=transport.pair(commands,folder,timeout=min(180,remaining))
    shares=[np.frombuffer((p/'index_shares.bin').read_bytes(),dtype='<u4') for p in paths]
    assert np.array_equal(shares[0]^shares[1],np.asarray(expected,dtype=np.uint32))
    assert np.array_equal(shares[0],np.array([int.from_bytes(data[0][48*i+44:48*i+48],'little') for i in range(201)]))
    parties=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
    assert parties[0]['sent']==parties[1]['received'] and parties[1]['sent']==parties[0]['received']
    row.update(name='alias_boundaries',all_correct=True,proposals_checked=201,parties=parties,
        total_wire_bytes=sum(p['sent'] for p in parties),
        files={str(p.relative_to(ROOT)):transport.sha(p) for folder in paths for p in folder.iterdir() if p.is_file()})
    records.append(row);print('alias zero/full/equality/highindex boundaries exact',flush=True)
    destination.write_text(json.dumps(dict(all_passed=True,results=records,input_sha256=spec['input_sha256'],
        preregistration_sha256=transport.sha(reg),limits=spec['limits']),indent=2)+'\n')


if __name__=='__main__':main()
