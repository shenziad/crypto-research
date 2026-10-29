"""Matched block costs and paired-window quality, without a novelty claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(median=float(np.median(a)),quartiles=np.quantile(a,[.25,.75]).tolist(),runs=len(a))


def main():
    results=ROOT/'results';hashes={};datasets={};costs={};comparisons={}
    def read(name):
        p=results/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    modes=('exact32','expanded33','compact25');variants=('selector','basis','weighted')
    for backend in ('cpu','cuda'):
        data=read(f'qwen_dithered_protocol_{backend}.json');datasets[backend]=data
        assert data['all_passed'] and len(data['results'])==162
        assert all(v['all_functional_checks_passed'] and v['process_context']=='spawn' for v in data['results'])
        costs[backend]={}
        for mode in modes:
            costs[backend][mode]={}
            for variant in variants:
                rows=[v for v in data['results'] if v['mode']==mode and v['variant']==variant]
                assert len(rows)==18
                r=costs[backend][mode][variant]={}
                for rtt in (0,50):
                    subset=[v for v in rows if v['emulated_rtt_ms']==rtt]
                    r[f'rtt{rtt}_ms']=summary([max(p['total_ms'] for p in v['parties']) for v in subset])
                r['dealer_ms']=summary([v['dealer_ms'] for v in rows])
                r['model_initialization_ms']=summary([max(p['initialization_ms'] for p in v['parties']) for v in rows])
                if backend=='cuda':r['context_upload_warmup_ms']=summary([max(p['cuda_setup_upload_ms'] for p in v['parties']) for v in rows])
                for out,key in [('key_array_bytes','total_key_array_bytes_both_parties'),('spline_key_array_bytes','spline_key_array_bytes_both_parties'),('online_bytes','online_sent_bytes_both_parties'),('model_initialization_bytes','model_initialization_sent_bytes_both_parties')]:
                    values=sorted(set(v[key] for v in rows));assert len(values)==1;r[out]=values[0]
                assert all(p['online_exchanges']==5 for v in rows for p in v['parties']);r['online_exchanges']=5
        comparisons[backend]={}
        for reference in ('exact32','expanded33'):
            comparisons[backend][reference]={}
            for variant in variants:
                ref=costs[backend][reference][variant];compact=costs[backend]['compact25'][variant]
                comparisons[backend][reference][variant]=dict(
                    online_rtt0_fraction_saved=1-compact['rtt0_ms']['median']/ref['rtt0_ms']['median'],
                    key_array_fraction_saved=1-compact['key_array_bytes']/ref['key_array_bytes'],
                    spline_array_fraction_saved=1-compact['spline_key_array_bytes']/ref['spline_key_array_bytes'],
                    same_function=reference=='expanded33',online_byte_saving=0)
    assert datasets['cpu']['capture_sha256']==datasets['cuda']['capture_sha256']
    quality={};paired={};rng=np.random.default_rng(20261029)
    for corpus,suffix in [('wiki32','_extended32'),('ptb32','')]:
        tail='_ptb32x256' if corpus=='ptb32' else ''
        prefix='qwen_public_spline_quality_f22'+suffix
        names={'aligned':prefix+'_alignedb8'+tail+'.json'}
        for seed in (20261029,20261030):names[f'dither_seed{seed}']=prefix+f'_ditherb8_seed{seed}'+tail+'.json'
        names['coarsebody']=prefix+'_ditherb8_seed20261029_coarsebody'+tail+'.json'
        files={k:read(v) for k,v in names.items() if (results/v).exists()}
        quality[corpus]={};paired[corpus]={};base=files['aligned'];offsets=[r['offset'] for r in base['records']]
        resample=rng.integers(0,len(offsets),(10000,len(offsets)))
        for label,d in files.items():
            assert [r['offset'] for r in d['records']]==offsets
            assert d['evaluation_corpus_sha256']==base['evaluation_corpus_sha256'] and d['range_sha256']==base['range_sha256']
            rows=d['records'];quality[corpus][label]=dict(predicted_positions=d['predicted_positions'],windows=len(rows),
                mean_kl=float(np.mean([r['mean_kl'] for r in rows])),top1_agreement=float(np.mean([r['top1_agreement'] for r in rows])),
                mean_nll_delta=float(np.mean([r['candidate_nll']-r['baseline_nll'] for r in rows])))
            if label=='aligned':continue
            diff={}
            for field in ('candidate_nll','mean_kl','top1_agreement'):
                delta=np.array([r[field]-b[field] for r,b in zip(rows,base['records'])])
                diff[field]=dict(mean=float(delta.mean()),paired_window_bootstrap95=np.quantile(delta[resample].mean(1),[.025,.975]).tolist())
            paired[corpus][label]=diff
    out=dict(costs=costs,compact_vs_reference=comparisons,quality=quality,quality_difference_from_aligned=paired,
        verified_full_block_runs=324,input_sha256=hashes,
        interpretation='Expanded33 andcompact25 implement the same public-randomized function. Exact32 is the aligned deterministic finite function; comparisons include approximation cost. No online communication reduction from classifier compression.',
        limitations='OneQwen0.5B model; finite24 gate/down numerical proxy, other operators float. Window bootstrap is descriptive on fixed corpora, not an independent model or full-corpus PPL guarantee. Local spawned parties share oneGPU; setup separately reported. Not upstream Sigma/FuseFSS/Grotto speed, complete joint security proof, or confirmed novelty.')
    (results/'dithered_protocol_report.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(3);width=.25
    for ax,backend in zip(axes,('cpu','cuda')):
        for j,mode in enumerate(modes):
            vals=[costs[backend][mode][v]['rtt0_ms']['median'] for v in variants]
            ax.bar(x+(j-1)*width,vals,width,label=mode)
        ax.set_xticks(x,variants);ax.set(ylabel='Median max-party online time (ms)',title=f'{backend.upper()}: finite 4864 x 896 block')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.savefig(results/'dithered_protocol_frontier.png',dpi=180)
    print(json.dumps(dict(comparison=comparisons,quality=quality,paired=paired),indent=2))


if __name__=='__main__':main()
