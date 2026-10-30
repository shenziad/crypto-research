"""Compile own GC harness with pinned existing toolchain; preserve old builds."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    build=ROOT/'third_party/FABLE/build'
    flags=(build/'CMakeFiles/private_row_harness.dir/flags.make').read_text().splitlines()
    arguments=[]
    for prefix in ('CXX_DEFINES = ', 'CXX_INCLUDES = ', 'CXX_FLAGS = '):
        arguments += shlex.split(next(s[len(prefix):] for s in flags if s.startswith(prefix)))
    link=shlex.split((build/'CMakeFiles/private_row_harness.dir/link.txt').read_text())
    outdir=ROOT/'toolchain/bounded-sampler';outdir.mkdir(exist_ok=True)
    binary=outdir/'bounded_accept_select'
    assert not binary.exists()
    source=ROOT/'experiments/bounded_accept_select.cpp'
    libraries=link[link.index('-o')+2:]
    command=['/usr/bin/g++',*arguments,str(source),'-o',str(binary),*libraries]
    subprocess.run(command,cwd=build,check=True)
    manifest=dict(command=command,cwd=str(build),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        limits='Own acceptance harness compiled against existing pinned SCI GC/dependencies. Existing FABLE sources and binaries unchanged.')
    (outdir/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),binary_sha256=manifest['binary_sha256'])))


if __name__=='__main__':main()
