import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def main():
    old=json.loads((ROOT/'toolchain/bounded-sampler/dense_adapter_build_manifest.json').read_bytes())
    source=ROOT/'experiments/qwen_selected_exp31.cpp';binary=ROOT/'toolchain/bounded-sampler/qwen_selected_exp31';assert not binary.exists()
    command=[str(source) if word==str(ROOT/'experiments/dense_adapter_linearot.cpp') else
        str(binary) if word==str(ROOT/'toolchain/bounded-sampler/dense_adapter_linearot') else word for word in old['command']]
    subprocess.run(command,cwd=old['cwd'],check=True)
    out=dict(command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (ROOT/'toolchain/bounded-sampler/qwen_selected_exp31_build_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(binary=str(binary),sha256=out['binary_sha256'])))
if __name__=='__main__':main()
