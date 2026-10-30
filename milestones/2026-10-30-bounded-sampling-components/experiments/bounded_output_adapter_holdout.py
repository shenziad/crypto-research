"""Fixed-weight followup after a preregistered bounded-adapter validation gate."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    registration = ROOT/'notes/20261030_bounded_output_adapter_holdout_preregistration.json'
    prereg = json.loads(registration.read_text())
    assert sha(Path(__file__).resolve()) == prereg['source_sha256']
    pilot_path = ROOT/'results/bounded_output_adapter_pilot.json'
    assert sha(pilot_path) == prereg['pilot_sha256']
    pilot = json.loads(pilot_path.read_text())
    assert pilot['pilot_quality_gate_passed']
    archive = ROOT/'results/bounded_output_adapter_weights.npz'
    assert sha(archive) == pilot['adapter_weights_sha256'] == prereg['weights_sha256']
    destination = ROOT/'results/bounded_output_adapter_holdout.json'
    assert not destination.exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = ROOT/'models/Pythia-14M'
    assert sha(folder/'model.safetensors') == pilot['model_sha256']
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1 << 30
    with np.load(archive) as arrays:
        adapters = {name: tuple(torch.tensor(arrays[name+'_'+part],device='cuda') for part in ('A','B'))
                    for name in ('linear','bounded')}
        scale = float(arrays['hidden_scale'])
    records = []
    b = pilot['bound']; fixed_k = pilot['worst_case_trials_for_failure_2neg40']
    for corpus, spec in prereg['corpora'].items():
        file = ROOT/'data'/spec['file']
        assert sha(file) == spec['sha256']
        tokens = tokenizer(file.read_text(),add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        for offset in spec['offsets']:
            ids = tokens[offset:offset+spec['length']][None].to('cuda')
            assert ids.shape[1] == spec['length']
            with torch.inference_mode():
                result = model(ids,use_cache=False,output_hidden_states=True)
                h = result.hidden_states[-1][0,:-1]
                z = result.logits[0,:-1].double(); target = ids[0,1:]
                row = dict(corpus=corpus,offset=offset,positions=len(target),
                    base_nll=float(torch.nn.functional.cross_entropy(z,target)))
                for name, (a, mat_b) in adapters.items():
                    raw = ((h/scale)@mat_b.T)@a.T
                    delta = b*torch.tanh(raw) if name=='bounded' else raw
                    delta=delta.double()
                    row[name+'_nll']=float(torch.nn.functional.cross_entropy(z+delta,target))
                    if name=='bounded':
                        assert float(delta.abs().max()) <= b
                        lp=z.log_softmax(-1);lpp=(z+delta).log_softmax(-1)
                        acceptance=(torch.logsumexp(z+delta,-1)-torch.logsumexp(z,-1)-b).exp()
                        assert float(acceptance.min()) >= math.exp(-2*b)-1e-12
                        row.update(mean_acceptance=float(acceptance.mean()),min_acceptance=float(acceptance.min()),
                            max_no_accept_probability=float(torch.exp(fixed_k*torch.log1p(-acceptance)).max()),
                            mean_kl_target_to_base=float((lpp.exp()*(lpp-lp)).sum(-1).mean()),
                            expected_distinct_base_proposals=float((-torch.expm1(fixed_k*torch.log1p(-lp.exp()))).sum(-1).mean()))
                records.append(row)
                print(corpus,offset,'deltaNLL',row['bounded_nll']-row['base_nll'],'accept',row['mean_acceptance'],flush=True)
    summary = {}
    for corpus in prereg['corpora']:
        rows=[r for r in records if r['corpus']==corpus]
        delta=np.array([r['bounded_nll']-r['base_nll'] for r in rows])
        control=np.array([r['linear_nll']-r['base_nll'] for r in rows])
        rng=np.random.default_rng(20261030)
        boot=delta[rng.integers(0,len(rows),(20000,len(rows)))].mean(1)
        summary[corpus]=dict(positions=sum(r['positions'] for r in rows),
            mean_bounded_minus_base_nll=float(delta.mean()),mean_linear_minus_base_nll=float(control.mean()),
            descriptive_window_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            mean_acceptance=float(np.mean([r['mean_acceptance'] for r in rows])),
            min_acceptance=min(r['min_acceptance'] for r in rows),
            max_no_accept_probability=max(r['max_no_accept_probability'] for r in rows))
    out=dict(status='completed',rank=pilot['rank'],bound=b,fixed_public_candidates=fixed_k,
        records=records,summary=summary,source_sha256=sha(Path(__file__).resolve()),
        preregistration_sha256=sha(registration),pilot_sha256=sha(pilot_path),weights_sha256=sha(archive),
        limits=prereg['limits'])
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
