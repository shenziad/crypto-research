"""Build own harness with header assertions; pinned compiled libraries unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=json.loads((ROOT/'toolchain/bounded-sampler/adapter_build_manifest.json').read_bytes())
    binary=ROOT/'toolchain/bounded-sampler/dense_adapter_linearot';assert not binary.exists()
    source=ROOT/'experiments/dense_adapter_linearot.cpp'
    command=[str(source) if word==str(ROOT/'experiments/bounded_adapter_gc.cpp') else
        str(binary) if word==str(ROOT/'toolchain/bounded-sampler/bounded_adapter_gc') else word
        for word in old['command']]
    command.insert(1,'-UNDEBUG')
    # The original -DNDEBUG occurs later: override it at the end too.
    command.append('-UNDEBUG')
    subprocess.run(command,cwd=old['cwd'],check=True)
    out=dict(command=command,cwd=old['cwd'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_source_sha256=hashlib.sha256((ROOT/'experiments/bounded_adapter_gc.cpp').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        limits='LinearOT products with accumulate=false, local ring sums, exact GC floors/PWL. This baseline stops at dense sharedscores; no complete privateweighted sampler yet.')
    (ROOT/'toolchain/bounded-sampler/dense_adapter_build_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),sha256=out['binary_sha256'])))


if __name__=='__main__':main()
