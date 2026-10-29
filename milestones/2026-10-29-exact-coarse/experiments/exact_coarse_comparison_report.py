"""Two-model exact/approximate classifier Pareto report from matched runs."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]


def main():
    costs={};hashes={};quality={};count=0
    for model in ('qwen05','qwen15'):
        costs[model]={}
        for backend in ('cpu','cuda'):
            name=f'qwen_exact_coarse_compare_{model}_{backend}.json';path=ROOT/'results'/name;data=json.loads(path.read_text());hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert data['all_passed'] and len(data['results'])==108;count+=len(data['results']);costs[model][backend]={}
            for mode in ('exact32','corrected24','compact25'):
                costs[model][backend][mode]={}
                for variant in ('basis','weighted'):
                    rows=[r for r in data['results'] if r['mode']==mode and r['variant']==variant]
                    assert len(rows)==18 and all(r['all_functional_checks_passed'] for r in rows)
                    assert all(r['gpu_capture_output_checked']==(mode!='compact25') for r in rows)
                    record={}
                    for rtt in (0,50):
                        values=[max(p['total_ms'] for p in r['parties']) for r in rows if r['emulated_rtt_ms']==rtt]
                        record[f'rtt{rtt}_ms']=dict(median=float(np.median(values)),quartiles=np.quantile(values,[.25,.75]).tolist(),runs=len(values))
                    for out,key in [('online_bytes','online_sent_bytes_both_parties'),('key_array_bytes','total_key_array_bytes_both_parties'),('spline_key_array_bytes','spline_key_array_bytes_both_parties'),('model_initialization_bytes','model_initialization_sent_bytes_both_parties')]:
                        values=set(r[key] for r in rows);assert len(values)==1;record[out]=values.pop()
                    assert all(p['online_exchanges']==5 for r in rows for p in r['parties']);record['online_exchanges']=5
                    record['dealer_ms_median']=float(np.median([r['dealer_ms'] for r in rows]));record['model_initialization_ms_median']=float(np.median([max(p['initialization_ms'] for p in r['parties']) for r in rows]))
                    if backend=='cuda':record['setup_upload_ms_median']=float(np.median([max(p['cuda_setup_upload_ms'] for p in r['parties']) for r in rows]))
                    costs[model][backend][mode][variant]=record
        quality[model]={}
        if model=='qwen05':
            names={'wiki32_aligned':'qwen_public_spline_quality_f22_extended32_alignedb8.json','ptb32_aligned':'qwen_public_spline_quality_f22_alignedb8_ptb32x256.json'}
        else:names={f'{corpus}{windows}_aligned':f'qwen15_dither_quality_aligned_b8_s20261029_{corpus}{windows}.json' for corpus in ('wiki','ptb') for windows in (4,32)}
        for label,name in names.items():
            p=ROOT/'results'/name
            if not p.exists():continue
            data=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();rows=data['records']
            quality[model][label]=dict(predicted_positions=data['predicted_positions'],windows=len(rows),mean_kl=float(np.mean([r['mean_kl'] for r in rows])),top1_agreement=float(np.mean([r['top1_agreement'] for r in rows])),mean_nll_delta=float(np.mean([r['candidate_nll']-r['baseline_nll'] for r in rows])))
    out=dict(costs=costs,aligned_function_quality=quality,verified_full_block_runs=count,input_sha256=hashes,
        exact_function='Exact32 andcorrected24 evaluate identical aligned finite functions at identical fullprecision A. Corrected24 uses one8-bit carry correction andan independently masked24-bit classifier. Compact25 is randomized andmay introduce selectionerror.',
        interpretation='Shorter classifier performance is achievable without randomized selection. Corrected24 adds8n bothparty bytes andretains5 exchanges forbasis/weighted. Approximation retains a material/traffic tradeoff, not a unique localcompute advantage.',
        limits='Two public pinned Qwen models, finite gate/down blocks only. SameGPU sequentialpartysharing; no realWAN,dealertransport orupstreamSigma/FuseFSS/Grotto comparisons. Setup separate. Standard TR/FSS components, independent novelty andcomplete jointsecurity notestablished.')
    (ROOT/'results/exact_coarse_comparison_report.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5));fig.subplots_adjust(bottom=.28,wspace=.28,top=.82);colors={'exact32':'#777777','corrected24':'#2379ae','compact25':'#df7d22'}
    for ax,model in zip(axes,('qwen05','qwen15')):
        for mode in ('exact32','corrected24','compact25'):
            for variant,marker in [('basis','o'),('weighted','s')]:
                r=costs[model]['cuda'][mode][variant];ax.scatter(r['online_bytes']/1024,r['rtt0_ms']['median'],color=colors[mode],marker=marker,s=65)
        ax.set(xlabel='Both-party online traffic (KiB)',ylabel='Median max-party online time (ms)',title='Qwen2.5-'+('0.5B' if model=='qwen05' else '1.5B'));ax.grid(alpha=.2);ax.margins(.1)
    handles=[Line2D([],[],color=color,marker='o',linestyle='',label=mode+(' (approx.)' if mode=='compact25' else '')) for mode,color in colors.items()]
    handles += [Line2D([],[],color='black',marker=marker,linestyle='',label=variant) for variant,marker in [('basis','o'),('weighted','s')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,frameon=False,fontsize=9)
    fig.text(.5,.015,'Each point: 9-run median. exact32/corrected24 share a deterministic function; compact25 randomizes selection.',ha='center',fontsize=8)
    fig.suptitle('CUDA finite gate/down blocks: same 5 exchanges; setup excluded',fontsize=12)
    fig.savefig(ROOT/'results/exact_coarse_pareto.png',dpi=180)
    fig.savefig(ROOT/'results/exact_coarse_pareto.pdf')
    print(json.dumps(dict(costs=costs,quality=quality),indent=2))


if __name__=='__main__':main()
