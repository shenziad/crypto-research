"""Two reused public model contexts for private arithmetic conformance only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    destination=ROOT/'results/bounded_adapter_gc_inputs.npz';assert not destination.exists()
    old=json.loads((ROOT/'notes/20261030_bounded_output_adapter_holdout_preregistration.json').read_bytes())
    pilot=json.loads((ROOT/'results/bounded_output_adapter_pilot.json').read_bytes())
    folder=ROOT/'models/Pythia-14M';assert sha(folder/'model.safetensors')==pilot['model_sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tokenizer.model_max_length=1<<30
    with np.load(ROOT/'results/bounded_output_adapter_weights.npz') as a:scale=float(a['hidden_scale'])
    xs=[];qs=[];cells=[]
    for corpus in ('ptb','wiki'):
        cell=old['corpora'][corpus];file=ROOT/'data'/cell['file'];assert sha(file)==cell['sha256']
        tokens=tokenizer(file.read_text(),add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        offset=cell['offsets'][0];ids=tokens[offset:offset+cell['length']][None].to('cuda')
        with torch.inference_mode():
            result=model(ids,use_cache=False,output_hidden_states=True)
            x=(result.hidden_states[-1][0,64]/scale).double().cpu().numpy()
            q=result.logits[0,64].double().softmax(-1).cpu().numpy()
        assert np.linalg.norm(x)<=20 and np.isfinite(q).all()
        xs.append(x);qs.append(q);cells.append(dict(corpus=corpus,offset=offset,position=64))
    np.savez_compressed(destination,x=np.array(xs),q=np.array(qs))
    manifest=dict(cells=cells,capture_sha256=sha(destination),source_sha256=sha(Path(__file__)),
        model_sha256=pilot['model_sha256'],weights_sha256=pilot['adapter_weights_sha256'],
        limits='Public-backbone inputs and proposal distributions from two previously evaluated contexts. Functional controls only; no new blind quality evidence, retraining, private score or sampler.')
    (ROOT/'results/bounded_adapter_gc_inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
