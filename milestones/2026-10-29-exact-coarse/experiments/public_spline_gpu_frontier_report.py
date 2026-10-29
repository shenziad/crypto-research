"""Verified CPU/CUDA full-block comparisons, excluding separately recorded setup."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=['qwen_public_spline_protocol_frontier_spawn.json','qwen_public_spline_protocol_frontier_spawn_cuda.json','qwen_public_spline_bandwidth_cuda.json']
    datasets=[json.loads((ROOT/'results'/f).read_text()) for f in files]
    assert all(v['all_passed'] for v in datasets)
    assert [len(v['results']) for v in datasets]==[96,96,54]
    assert all(v['all_functional_checks_passed'] and v['gpu_capture_output_checked'] for d in datasets for v in d['results'])
    assert all(v['process_context']=='spawn' for d in datasets[:2] for v in d['results'])
    assert datasets[0]['capture_sha256']==datasets[1]['capture_sha256']
    variants=('selector','basis','weighted','wide');result=dict(protocols={},verified_full_block_runs=246,
        hardware=json.loads((ROOT/'results/dcf_cuda_check.json').read_text())['hardware'],
        input_hashes={f:hashlib.sha256((ROOT/'results'/f).read_bytes()).hexdigest() for f in files},
        scope='Full finite B rounding/clipping, public SiLU FSS, gated multiplication, private down matrix and final signed shift. Two spawned processes share one physical GPU. Upstream projections and normalized encoding excluded.',
        timing='Online wall clock includes public query construction, H2D/D2H, kernel synchronization and sockets. CUDA context/key/input upload/warmup and model initialization are separately recorded, not zero cost.',
        prior='Do not claim new primitive or upstream Sigma/FuseFSS superiority.2021 already optimizes publicly scalar-related spline payloads; this is a standard-linearity application candidate.')
    for variant in variants:
        rec={}
        for label,d in zip(('cpu','cuda'),datasets[:2]):
            subset=[v for v in d['results'] if v['variant']==variant]
            for rtt in (0,50):
                values=[max(p['total_ms'] for p in v['parties']) for v in subset if v['emulated_rtt_ms']==rtt]
                rec[f'{label}_rtt{rtt}_both_party_max_ms']=dict(median=float(np.median(values)),quartiles=np.quantile(values,[.25,.75]).tolist(),runs=len(values))
            rec[f'{label}_model_initialization_ms_median']=float(np.median([max(p['initialization_ms'] for p in v['parties']) for v in subset]))
            if label=='cuda':rec['cuda_context_upload_warmup_ms_median']=float(np.median([max(p['cuda_setup_upload_ms'] for p in v['parties']) for v in subset]))
        subset=[v for v in datasets[1]['results'] if v['variant']==variant]
        rec['online_bytes_both_parties']=subset[0]['online_sent_bytes_both_parties'];rec['exchanges']=5
        rec['key_array_bytes_range_both_parties']=[min(v['total_key_array_bytes_both_parties'] for v in subset),max(v['total_key_array_bytes_both_parties'] for v in subset)]
        rec['model_initialization_bytes_both_parties']=subset[0]['model_initialization_sent_bytes_both_parties']
        for rate in (10,100):
            subset=[v for v in datasets[2]['results'] if v['variant']==variant and v['payload_delay_mbps_per_direction']==rate]
            if subset:rec[f'cuda_payload_delay_{rate}Mbps_rtt20_both_party_max_ms']=float(np.median([max(p['total_ms'] for p in v['parties']) for v in subset]))
        result['protocols'][variant]=rec
    (ROOT/'results/public_spline_gpu_frontier_report.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(4);width=.34
    for j,label in enumerate(('cpu','cuda')):
        values=[result['protocols'][v][f'{label}_rtt0_both_party_max_ms']['median'] for v in variants]
        axes[0].bar(x+(j-.5)*width,values,width,label=label)
    axes[0].set_xticks(x,variants);axes[0].set(ylabel='Median online wall clock (ms)',title='Same full finite block; spawned parties');axes[0].legend(frameon=False)
    x=np.arange(2);width=.24
    for j,(variant,color) in enumerate(zip(variants[:3],('#777777','#2676b5','#db6f14'))):
        values=[result['protocols'][variant][f'cuda_payload_delay_{rate}Mbps_rtt20_both_party_max_ms'] for rate in (10,100)]
        axes[1].bar(x+(j-1)*width,values,width,label=variant,color=color)
    axes[1].set_xticks(x,['10Mbps','100Mbps']);axes[1].set(ylabel='Median online wall clock (ms)',title='CUDA +payload-delay model +20ms/exchange');axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'results/public_spline_gpu_frontier.png',dpi=180);fig.savefig(ROOT/'results/public_spline_gpu_frontier.pdf')
    print(json.dumps(result['protocols'],indent=2))


if __name__=='__main__':main()
