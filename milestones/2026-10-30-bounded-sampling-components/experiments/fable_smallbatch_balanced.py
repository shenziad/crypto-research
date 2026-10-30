"""Four alternating paired full-protocol measurements on matched frozen rows."""
import json
import os
from pathlib import Path
import statistics

import numpy as np
import run_fable_smallbatch_check as harness

ROOT = Path(__file__).resolve().parents[1]


def main():
    registration = ROOT/'notes/20261030_fable_smallbatch_balanced_preregistration.json'
    spec = json.loads(registration.read_bytes())
    destination = ROOT/'results/fable_smallbatch_balanced.json'
    assert not destination.exists()
    for name, digest in spec['input_sha256'].items():
        assert harness.sha(ROOT/name) == digest, name
    old = ROOT/'results/fable_private_row_check_runs/frozen_A_191_plus65'
    table = np.frombuffer((old/'owner_table.bin').read_bytes(), dtype=np.uint8).reshape(65536, 32)
    indices = np.bitwise_xor(*[np.frombuffer((old/name).read_bytes(), dtype='<u2') for name in
        ('owner_index_shares.bin', 'other_index_shares.bin')])
    assert len(indices) == 256
    folder = ROOT/'results/fable_smallbatch_balanced_runs'
    folder.mkdir()
    os.environ['OMP_NUM_THREADS'] = '4'
    rows = []
    for repetition, order in enumerate(spec['orders']):
        for variant in order:
            sub = folder/f'{repetition}_{variant}'
            sub.mkdir()
            source = ROOT/spec['variants'][variant]
            harness.SOURCE = source
            tablefile = sub/'owner_table.bin'
            tablefile.write_bytes(table.tobytes())
            owner = np.frombuffer(os.urandom(512), dtype='<u2')
            other = owner ^ indices
            inputs = [sub/'owner_index_shares.bin', sub/'other_index_shares.bin']
            for path, value in zip(inputs, (owner, other)):
                path.write_bytes(value.tobytes())
            outputs = [sub/f'party{i+1}_row_shares.bin' for i in range(2)]
            metrics = [sub/f'party{i+1}_metrics.json' for i in range(2)]
            port = harness.port()
            commands = [[str(source/'build/bin/private_row_harness'), str(i+1), str(port), '256',
                str(tablefile) if i == 0 else '-', str(inputs[i]), str(outputs[i]), str(metrics[i]), '4']
                for i in range(2)]
            row = harness.pair(commands, sub)
            shares = [np.frombuffer(p.read_bytes(), dtype=np.uint8).reshape(256, 32) for p in outputs]
            assert np.array_equal(shares[0] ^ shares[1], table[indices])
            parties = [json.loads(p.read_bytes()) for p in metrics]
            assert parties[0]['sent'] == parties[1]['received']
            assert parties[1]['sent'] == parties[0]['received']
            row.update(repetition=repetition, variant=variant, all_bytes_exact=True,
                parties=parties, total_wire_bytes=sum(x['sent'] for x in parties),
                max_party_protocol_ms=max(x['protocol_ms'] for x in parties))
            rows.append(row)
            print(variant, repetition, row['total_wire_bytes'], row['max_party_protocol_ms'], flush=True)
    summary = {variant:dict(observations=4,
        median_max_party_protocol_ms=statistics.median(r['max_party_protocol_ms'] for r in rows if r['variant'] == variant),
        median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in rows if r['variant'] == variant))
        for variant in spec['variants']}
    out = dict(all_passed=True, exact_rows_checked=8*256, results=rows, summary=summary,
        preregistration_sha256=harness.sha(registration), input_sha256=spec['input_sha256'], limits=spec['limits'])
    destination.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
