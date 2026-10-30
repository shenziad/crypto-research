"""Registered standalone placement checks; no protocol processes or model changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reference(tag, slots):
    rank = int.from_bytes(tag, 'little') % (slots * (slots - 1) * (slots - 2))
    # Independent rank-deletion implementation; avoids the C++ skip arithmetic.
    remaining = list(range(slots))
    out = []
    for radix in (slots, slots - 1, slots - 2):
        out.append(remaining.pop(rank % radix))
        rank //= radix
    assert rank == 0
    return out


def main():
    reg = ROOT/'notes/20261031_private_lut_placement_preregistration.json'
    spec = json.loads(reg.read_bytes())
    start = datetime.fromisoformat(spec['created_utc'])
    def remaining():
        value = spec['maximum_wall_seconds'] - (datetime.now(timezone.utc) - start).total_seconds()
        assert value > 0, 'placement stage budget exhausted'
        return value
    for name, expected in spec['input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    destination = ROOT/'results/private_lut_placement_check.json'
    assert not destination.exists()
    folder = ROOT/'results/private_lut_placement_check_runs'
    folder.mkdir()
    binary = ROOT/'toolchain/bounded-sampler/private_lut_placement_check'
    command = ['/usr/bin/g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               str(ROOT/'experiments/private_lut_placement_check.cpp'), '-o', str(binary)]
    subprocess.run(command, check=True, timeout=remaining(), cwd=ROOT)
    N = spec['fullsize_unique_ids']
    # Public deterministic correctness fixture, never a protocol deployment key.
    key = hashlib.sha256(b'placement correctness fixture 20261031').digest()[:16]
    plain = bytearray()
    for i in range(N):
        plain.extend(i.to_bytes(16, 'little'))
        plain.extend((i + (1 << 127)).to_bytes(16, 'little'))
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    tags = enc.update(bytes(plain)) + enc.finalize()
    tag_path = folder/'tags.bin'; tag_path.write_bytes(tags)
    mapped_path = folder/'mapped.bin'; metrics_path = folder/'native_metrics.json'
    rng = random.Random(spec['seed'])
    cases = []
    for slots in (3, 4, 5, 32, 2048, 4096):
        d = slots * (slots-1) * (slots-2)
        for value in [0, 1, d-1, d, d+1, (1<<127)-1, 1<<127, (1<<128)-1] + [rng.getrandbits(128) for _ in range(1024)]:
            cases.append((slots, value.to_bytes(16, 'little')))
    case_path = folder/'cases.bin'; case_maps = folder/'case_maps.bin'
    case_path.write_bytes(b''.join(s.to_bytes(4, 'little')+v for s,v in cases))
    begin = time.monotonic()
    subprocess.run([str(binary), str(tag_path), str(mapped_path), str(metrics_path), str(case_path), str(case_maps)],
                   check=True, timeout=remaining(), cwd=ROOT)
    native = json.loads(metrics_path.read_bytes())
    mapped = np.frombuffer(mapped_path.read_bytes(), dtype='<u4').reshape(N, 6)
    for i in range(N):
        for j, slots in enumerate((2048, 4096)):
            ref = reference(tags[32*i+16*j:32*i+16*j+16], slots)
            assert mapped[i, 3*j:3*j+3].tolist() == ref, (i, j)
        if i % 4096 == 0:
            remaining()
    actual_cases = np.frombuffer(case_maps.read_bytes(), dtype='<u4').reshape(len(cases), 3)
    for i, (slots, tag) in enumerate(cases):
        assert actual_cases[i].tolist() == reference(tag, slots), i
    completed = datetime.now(timezone.utc)
    elapsed = (completed-start).total_seconds(); assert elapsed < spec['maximum_wall_seconds']
    paths = [reg, ROOT/'experiments/private_lut_placement.h',
             ROOT/'experiments/private_lut_placement_check.cpp', Path(__file__),
             binary, tag_path, mapped_path, metrics_path, case_path, case_maps]
    result = dict(all_passed=True, native=native, all_fullsize_mappings_independently_checked=2*N,
                  fixed_public_AES_key_hex=key.hex(), AES_domain_definition='AES(key, LE128(id + domain*2^127)); domains0,1; id<2^19',
                  native_and_reference_wall_seconds=time.monotonic()-begin,
                  created_utc=spec['created_utc'], completed_utc=completed.isoformat(),
                  stage_wall_upper_seconds=elapsed, maximum_wall_seconds=spec['maximum_wall_seconds'],
                  two_party_pairs=0, fullsize_fixed_key_tables=1, retries=0,
                  source_and_fixture_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
                  limits=spec['limits'])
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_passed','native','all_fullsize_mappings_independently_checked','stage_wall_upper_seconds')},indent=2))


if __name__ == '__main__':
    main()
