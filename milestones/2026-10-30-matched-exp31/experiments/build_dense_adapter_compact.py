"""Build an exactly equivalent public-PWL circuit and its plaintext core check."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=json.loads((ROOT/'toolchain/bounded-sampler/dense_adapter_build_manifest.json').read_bytes())
    outputs={}
    for stem in ('public_tanh_compact_check','dense_adapter_compact'):
        binary=ROOT/'toolchain/bounded-sampler'/stem;assert not binary.exists()
        source=ROOT/'experiments'/(stem+'.cpp')
        command=[str(source) if word==str(ROOT/'experiments/dense_adapter_linearot.cpp') else
            str(binary) if word==str(ROOT/'toolchain/bounded-sampler/dense_adapter_linearot') else word
            for word in old['command']]
        subprocess.run(command,cwd=old['cwd'],check=True)
        outputs[stem]=dict(command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    out=dict(outputs=outputs,header_sha256=hashlib.sha256((ROOT/'experiments/public_tanh_compact.h').read_bytes()).hexdigest(),
             limits='Standard Boolean circuit engineering only; exact same finite PWL function and private score output contract. No new primitive or precision/model change.')
    (ROOT/'toolchain/bounded-sampler/dense_compact_build_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({n:v['binary_sha256'] for n,v in outputs.items()}))


if __name__=='__main__':main()
