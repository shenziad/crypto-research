"""Build the supplementary matched-exp31 dense sampler and CDF controls."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def main():
    old=json.loads((ROOT/'toolchain/bounded-sampler/dense_adapter_build_manifest.json').read_bytes())
    out={}
    for stem in ('private_weighted_cdf95_check','qwen_dense_cdf31'):
        source=ROOT/'experiments'/(stem+'.cpp');binary=ROOT/'toolchain/bounded-sampler'/stem;assert not binary.exists()
        command=[str(source) if word==str(ROOT/'experiments/dense_adapter_linearot.cpp') else
            str(binary) if word==str(ROOT/'toolchain/bounded-sampler/dense_adapter_linearot') else word for word in old['command']]
        subprocess.run(command,cwd=old['cwd'],check=True)
        out[stem]=dict(command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (ROOT/'toolchain/bounded-sampler/qwen_dense_cdf31_build_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({n:x['binary_sha256'] for n,x in out.items()}))
if __name__=='__main__':main()
