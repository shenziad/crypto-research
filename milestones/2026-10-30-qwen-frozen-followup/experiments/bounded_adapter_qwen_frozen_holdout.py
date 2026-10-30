"""Prospectively registered frozen-weight follow-up after a validation gate miss.

The previous50% gate result is unchanged. This separate study is motivated by
positive absolute validation improvement and the user's request to continue.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    registration=ROOT/'notes/20261030_qwen_frozen_holdout_preregistration.json'
    spec=json.loads(registration.read_bytes())
    assert sha(Path(__file__))==spec['source_sha256']
    for n,digest in spec['input_sha256'].items():
        assert sha(ROOT/n)==digest,n
    pilot=json.loads((ROOT/spec['pilot']).read_bytes())
    assert not pilot['validation_gate_passed'] and not pilot['holdout']
    destination=ROOT/'results/bounded_adapter_qwen_frozen_holdout.json'
    assert not destination.exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'models/Qwen2.5-0.5B'
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    model.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tokenizer.model_max_length=1<<30
    archive=ROOT/'results/bounded_adapter_qwen_transfer_weights.npz'
    with np.load(archive) as arrays:
        adapters={name:tuple(torch.tensor(arrays[name+'_'+part],device='cuda') for part in ('A','B'))
                  for name in ('linear','bounded')}
        scale=float(arrays['hidden_scale'])
    assert scale==pilot['hidden_scale']
    records=[];started=time.monotonic();b=1.;k=201
    captures={}
    for corpus,item in spec['corpora'].items():
        path=ROOT/'data'/item['file'];assert sha(path)==item['sha256']
        tokens=tokenizer(path.read_text(),add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        for offset in item['offsets']:
            ids=tokens[offset:offset+129][None].to('cuda');assert ids.shape==(1,129)
            with torch.inference_mode():
                out=model(ids,use_cache=False,output_hidden_states=True)
                h=out.hidden_states[-1][0,:-1]
                z=out.logits[0,:-1].double();target=ids[0,1:]
                row=dict(corpus=corpus,offset=offset,positions=len(target),
                    base_nll=float(torch.nn.functional.cross_entropy(z,target)),
                    max_normalized_input_l2=float(torch.linalg.vector_norm(h/scale,dim=-1).max()))
                for name,(a,mat_b) in adapters.items():
                    raw=((h/scale)@mat_b.T)@a.T
                    delta=(torch.tanh(raw) if name=='bounded' else raw).double()
                    row[name+'_nll']=float(torch.nn.functional.cross_entropy(z+delta,target))
                    row[name+'_max_abs_correction']=float(delta.abs().max())
                    if name=='bounded':
                        assert float(delta.abs().max())<=b
                        lp=z.log_softmax(-1);lpp=(z+delta).log_softmax(-1)
                        acceptance=(torch.logsumexp(z+delta,-1)-torch.logsumexp(z,-1)-b).exp()
                        assert float(acceptance.min())>=math.exp(-2*b)-1e-12
                        assert float(acceptance.max())<=1+1e-12
                        row.update(mean_acceptance=float(acceptance.mean()),
                            min_acceptance=float(acceptance.min()),
                            max_no_accept_probability=float(torch.exp(k*torch.log1p(-acceptance)).max()),
                            mean_kl_target_to_base=float((lpp.exp()*(lpp-lp)).sum(-1).mean()))
                if offset==item['offsets'][0]:
                    # Fixed predeclared position64. Raw inputs stay on NEWSMY.
                    captures[corpus+'_x']=(h[64]/scale).cpu().numpy()
                    captures[corpus+'_q']=z[64].softmax(-1).cpu().numpy()
                records.append(row)
                print(corpus,offset,'bounded_minus_base',row['bounded_nll']-row['base_nll'],
                      'linear_minus_base',row['linear_nll']-row['base_nll'],flush=True)
    summary={}
    for corpus in spec['corpora']:
        rows=[r for r in records if r['corpus']==corpus]
        d=np.array([r['bounded_nll']-r['base_nll'] for r in rows])
        c=np.array([r['linear_nll']-r['base_nll'] for r in rows])
        rng=np.random.default_rng(20261030)
        boot=d[rng.integers(0,len(rows),(20000,len(rows)))].mean(1)
        summary[corpus]=dict(positions=sum(r['positions'] for r in rows),
            base_nll=float(np.mean([r['base_nll'] for r in rows])),
            mean_bounded_minus_base_nll=float(d.mean()),mean_linear_minus_base_nll=float(c.mean()),
            retained_linear_NLL_gain=float(d.mean()/c.mean()) if c.mean()<0 else None,
            descriptive_reset_window_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            mean_acceptance=float(np.mean([r['mean_acceptance'] for r in rows])),
            min_acceptance=min(r['min_acceptance'] for r in rows),
            max_no_accept_probability=max(r['max_no_accept_probability'] for r in rows),
            max_normalized_input_l2=max(r['max_normalized_input_l2'] for r in rows))
    capture_path=ROOT/'results/bounded_adapter_qwen_frozen_captures.npz'
    assert not capture_path.exists();np.savez_compressed(capture_path,**captures)
    # A prospective diagnostic; it does not replace the original50% gate.
    positive=all(s['mean_bounded_minus_base_nll']<0 and
                 s['descriptive_reset_window_bootstrap95'][1]<0 for s in summary.values())
    result=dict(status='completed',summary=summary,records=records,
        positive_crosscorpus_diagnostic=positive,
        original_validation_gate_still_failed=True,
        no_training_or_hyperparameter_changes=True,rank=8,bound=b,fixed_candidates=k,
        weights_sha256=sha(archive),captures_sha256=sha(capture_path),
        source_sha256=sha(Path(__file__)),preregistration_sha256=sha(registration),
        input_sha256=spec['input_sha256'],device=torch.cuda.get_device_name(),
        elapsed_seconds_diagnostic=time.monotonic()-started,limits=spec['limits'])
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,positive_crosscorpus_diagnostic=positive),indent=2),flush=True)


if __name__=='__main__':
    main()
