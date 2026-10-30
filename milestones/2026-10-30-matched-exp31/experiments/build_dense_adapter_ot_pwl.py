"""Build the standard-OT exact PWL variant, preserving both GC baselines."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=json.loads((ROOT/'toolchain/bounded-sampler/dense_adapter_build_manifest.json').read_bytes())
    binary=ROOT/'toolchain/bounded-sampler/dense_adapter_ot_pwl';assert not binary.exists()
    source=ROOT/'experiments/dense_adapter_ot_pwl.cpp'
    command=[str(source) if word==str(ROOT/'experiments/dense_adapter_linearot.cpp') else
        str(binary) if word==str(ROOT/'toolchain/bounded-sampler/dense_adapter_linearot') else word
        for word in old['command']]
    subprocess.run(command,cwd=old['cwd'],check=True)
    out=dict(command=command,cwd=old['cwd'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256((ROOT/'experiments/public_tanh_ot.h').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        limits='ExistingSCI arithmetic/MSB/mux/LUT/extension functions, exact same frozen PWL; no new primitive or precisionchange. Dense scores only, fullsampler incomplete.')
    (ROOT/'toolchain/bounded-sampler/dense_ot_pwl_build_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),sha256=out['binary_sha256'])))


if __name__=='__main__':main()
