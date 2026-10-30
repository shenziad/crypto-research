"""Build one own alias-finish harness; all earlier binaries stay frozen."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    previous=ROOT/'toolchain/bounded-sampler/adapter_build_manifest.json'
    old=json.loads(previous.read_bytes())
    binary=ROOT/'toolchain/bounded-sampler/bounded_alias_batch';assert not binary.exists()
    source=ROOT/'experiments/bounded_alias_batch.cpp'
    command=[str(source) if word==str(ROOT/'experiments/bounded_adapter_gc.cpp') else
             str(binary) if word==str(ROOT/'toolchain/bounded-sampler/bounded_adapter_gc') else word
             for word in old['command']]
    subprocess.run(command,cwd=old['cwd'],check=True)
    manifest=dict(command=command,cwd=old['cwd'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        previous_build_manifest_sha256=hashlib.sha256(previous.read_bytes()).hexdigest())
    (ROOT/'toolchain/bounded-sampler/alias_batch_build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),sha256=manifest['binary_sha256'])))


if __name__=='__main__':main()
