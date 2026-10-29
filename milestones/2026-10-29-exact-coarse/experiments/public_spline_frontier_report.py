"""Summarize measured cost frontier, quality and incremental offline tradeoffs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(20261029);rows=json.loads((ROOT/'results/qwen_public_spline_protocol_frontier.json').read_text())['results']
    bandwidth=json.loads((ROOT/'results/qwen_public_spline_bandwidth.json').read_text())['results']
    out=dict(scope='One4864x896 finite gate/down block, both-party bytes,5 exchanges, private W initialization69730304B separate; no fullLLM latency claim.',protocols={},quality={})
    variants=['selector','basis','weighted','wide']
    for variant in variants:
        subset=[v for v in rows if v['variant']==variant];first=subset[0]
        rec=dict(online_bytes=first['online_sent_bytes_both_parties'],
            key_array_bytes_min=min(v['total_key_array_bytes_both_parties'] for v in subset),
            key_array_bytes_max=max(v['total_key_array_bytes_both_parties'] for v in subset),exchanges=5)
        for rtt in (0,50):
            t=[v['parties'][0]['total_ms'] for v in subset if v['emulated_rtt_ms']==rtt]
            rec[f'party0_rtt{rtt}_ms']=dict(median=float(np.median(t)),quartiles=np.quantile(t,[.25,.75]).tolist(),runs=len(t))
        for rate in (10,100):
            t=[v['parties'][0]['total_ms'] for v in bandwidth if v['variant']==variant and v['payload_delay_mbps_per_direction']==rate]
            if t:rec[f'payload_delay_{rate}Mbps_rtt20_ms']=float(np.median(t))
        out['protocols'][variant]=rec
    for candidate,baseline in [('basis','selector'),('weighted','selector'),('weighted','basis')]:
        a=out['protocols'][candidate];b=out['protocols'][baseline]
        extra=a['key_array_bytes_min']-b['key_array_bytes_min'];saved=b['online_bytes']-a['online_bytes']
        out[f'{candidate}_vs_{baseline}']=dict(online_bytes_saved=saved,online_fraction_saved=saved/b['online_bytes'],
            extra_material_bytes=extra,extra_material_fraction=extra/b['key_array_bytes_min'],
            incremental_offline_online_byte_ratio=extra/saved,
            interpretation='Array bytes are not measured dealer transport. If sent literally, the added offline bytes/saved online bytes gives a necessary bandwidth-ratio threshold before CPU cost. No mask/key reuse across requests is permitted by these runs.')
    for label,suffix in [('wikitext_4','f22'),('wikitext_32','f22_extended32'),('ptb_32','f22_ptb32x256')]:
        path=ROOT/'results'/f'qwen_public_spline_quality_{suffix}.json';d=json.loads(path.read_text());records=d['records']
        delta=np.asarray([v['candidate_nll']-v['baseline_nll'] for v in records]);idx=rng.integers(len(delta),size=(10000,len(delta)))
        out['quality'][label]=dict(predicted_positions=d['predicted_positions'],windows=len(records),
            mean_kl=float(np.mean([v['mean_kl'] for v in records])),top1_agreement=float(np.mean([v['top1_agreement'] for v in records])),
            mean_nll_delta=float(delta.mean()),nll_delta_window_bootstrap95=np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist(),
            uncertainty='Descriptive resampling of windows in one fixed corpus/model, not an independent model/generalization confidence guarantee.',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    out['input_hashes']={f:hashlib.sha256((ROOT/'results'/f).read_bytes()).hexdigest() for f in ['qwen_public_spline_protocol_frontier.json','qwen_public_spline_bandwidth.json']}
    (ROOT/'results/public_spline_frontier_report.json').write_text(json.dumps(out,indent=2)+'\n')
    colors=['#666666','#2676b5','#db6f14','#9a9a9a'];fig,axes=plt.subplots(1,2,figsize=(10,4.0),layout='constrained')
    for v,color in zip(variants,colors):
        rec=out['protocols'][v];axes[0].scatter(rec['key_array_bytes_min']/1e6,rec['online_bytes']/1000,s=70,color=color,label=v)
        axes[0].annotate(v,(rec['key_array_bytes_min']/1e6,rec['online_bytes']/1000),xytext=(6,5),textcoords='offset points',fontsize=9)
    axes[0].set(xlabel='Preprocessing arrays, both parties (MB)',ylabel='Online sent, both parties (kB)',title='Same finite function; all5 exchanges')
    axes[0].set_xlim(96,240);axes[0].set_ylim(310,497)
    axes[0].grid(alpha=.2)
    x=np.arange(2);width=.24
    for j,(v,color) in enumerate(zip(variants[:3],colors[:3])):
        rec=out['protocols'][v];times=[rec[f'payload_delay_{rate}Mbps_rtt20_ms'] for rate in (10,100)]
        axes[1].bar(x+(j-1)*width,times,width,color=color,label=v)
    axes[1].set_xticks(x,['10Mbps','100Mbps']);axes[1].set(ylabel='Median party0 online (ms)',title='Payload-delay model +20ms/exchange')
    axes[1].legend(frameon=False);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results/public_spline_frontier.png',dpi=180)
    fig.savefig(ROOT/'results/public_spline_frontier.pdf')
    print(json.dumps({k:v for k,v in out.items() if k in ['weighted_vs_selector','quality']},indent=2))


if __name__=='__main__':main()
