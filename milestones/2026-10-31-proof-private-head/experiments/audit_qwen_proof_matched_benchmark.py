"""After-exit saved-file replay; retains noisy wall observations without filtering."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from finite_alias_table import build
from run_dense_adapter_linearot import reconstruct
from run_selected_adapter_exp31 import expected
from run_private_weighted_cdf95_check import audit as cdf_audit
from run_qwen_proof_sampling import audit as proof_audit
from fable_proof_integration_budget import verify_ready

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if argv and Path(argv[0].decode(errors='replace')).name in {'qwen_dense_cdf31','qwen_selected_exp31','qwen_alias_batch','private_row_harness'}:
            active.append(str(p.parent))
    assert not active,active
    verify_ready()
    regp=ROOT/'notes/20261031_qwen_proof_benchmark_preregistration.json'
    reg=json.loads(regp.read_bytes());rp=ROOT/'results/qwen_proof_matched_benchmark.json'
    result=json.loads(rp.read_bytes());assert result['all_passed'] and result['preregistration_sha256']==sha(regp)
    verified={}
    def check(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('input_sha256','files','audit_file_sha256','verified_files'):
                    for n,h in item.items():
                        assert sha(ROOT/n)==h,n
                        if n in verified:assert verified[n]==h
                        verified[n]=h
                else:check(item)
        elif isinstance(value,list):
            for item in value:check(item)
    check(reg);check(result)
    common=ROOT/'results/qwen_proof_matched_benchmark_runs/common'
    with np.load(common/'client.npz') as z:x=z['x_integer'];q=z['q']
    with np.load(common/'server.npz') as z:a=z['a_integer'];b=z['b_integer']
    mass=build(q,44)[2][:151936].astype(np.uint64);assert sum(map(int,mass))==1<<62
    full=expected(x,b,a)
    assert sha(common/'client.npz')==reg['input_sha256']['results/qwen_proof_sampling_runs/ptb_test/client_initial.npz']
    assert hashlib.sha256(mass.astype('<u8').tobytes()).hexdigest()==result['target_q_mass_sha256']
    assert hashlib.sha256(full[-1].astype('<u8').tobytes()).hexdigest()==result['target_E31_vector_sha256']
    seen=[]
    for row in result['results']:
        folder=common.parent/f"{row['repetition']}_{row['variant']}"
        if row['variant']=='proof':
            replay=proof_audit(folder,x,b,a);assert replay['total_wire_bytes']==row['total_wire_bytes']
            with np.load(folder/'party0/client_alias_audit.npz') as z:assert np.array_equal(z['mass'][:151936],mass)
        else:
            paths=[folder/f'party{i}' for i in (1,2)]
            for name,width,value in zip(('raw_t_add41.bin','t_add26.bin','raw_z_add44.bin','delta_add44.bin','exp_add33.bin'),(41,26,44,44,33),full):
                assert np.array_equal(reconstruct(paths,name,width),value),name
            cdf_audit(folder,mass.tolist(),full[-1].tolist())
            party=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
            for c in range(3):
                assert party[0]['end_counters'][c]==party[1]['end_counters'][c+3]
                assert party[1]['end_counters'][c]==party[0]['end_counters'][c+3]
            assert sum(sum(p['end_counters'][:3]) for p in party)==row['total_wire_bytes']
        seen.append(row['variant'])
    assert seen==[v for order in reg['orders'] for v in order]
    summary={}
    wall_ranges={}
    for v in ('dense','proof'):
        rows=[r for r in result['results'] if r['variant']==v];assert len(rows)==4
        summary[v]=dict(observations=4,median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in rows),
            median_whole_pair_wall_ms=statistics.median(r['whole_pair_wall_ms'] for r in rows),
            median_participant_elapsed_ms=statistics.median(r['participant_elapsed_ms'] for r in rows))
        wall_ranges[v]=dict(min_ms=min(r['whole_pair_wall_ms'] for r in rows),max_ms=max(r['whole_pair_wall_ms'] for r in rows))
    assert summary==result['summary']
    pairs=[]
    for rep in range(4):
        rr={r['variant']:r for r in result['results'] if r['repetition']==rep}
        pairs.append(dict(repetition=rep,proof_over_dense_wall=rr['proof']['whole_pair_wall_ms']/rr['dense']['whole_pair_wall_ms']))
    lp=ROOT/'results/qwen_proof_benchmark_ledger.json';ledger=json.loads(lp.read_bytes())
    assert ledger['native_pairs_reserved']==20<=reg['maximum_native_two_party_pairs']
    assert len(ledger['events'])==8 and all(e.get('success') and 'completed_utc' in e for e in ledger['events'])
    end=datetime.now(timezone.utc);elapsed=(end-datetime.fromisoformat(reg['created_utc'])).total_seconds();assert elapsed<reg['maximum_wall_seconds']
    ledger.update(stage_closed=True,actual_native_pairs=20,completed_utc=end.isoformat(),stage_wall_upper_seconds=elapsed);lp.write_text(json.dumps(ledger,indent=2)+'\n')
    out=dict(all_passed=True,retrospective=True,summary=summary,
        communication_reduction_percent=100*(1-summary['proof']['median_total_wire_bytes']/summary['dense']['median_total_wire_bytes']),
        primary_median_wall_reduction_percent=100*(1-summary['proof']['median_whole_pair_wall_ms']/summary['dense']['median_whole_pair_wall_ms']),
        wall_ranges=wall_ranges,paired_wall_ratios_descriptive=pairs,
        classification='communication_positive_wall_unstable_no_general_speedup_claim',
        complete_benchmark_runs_replayed=8,benchmark_native_pairs=20,native_failures=0,native_retries=0,
        registered_files_verified=len(verified),verified_files=verified,
        stage_wall_upper_seconds=elapsed,maximum_wall_seconds=reg['maximum_wall_seconds'],
        result_sha256=sha(rp),preregistration_sha256=sha(regp),source_sha256=sha(Path(__file__)),
        limits='All8runsretained; resourcesnapshotsrecordCPUaround0.54..1.48GHz duringpartofrunandzeromemorypressure. Causalitynotestablished; noexclusiveCPU/controlledfrequencyclaim. Mediansandrangesaredescriptive; no discardedoutliers/significancetest/WAN/SOTAclaim. Securityfromseparateassumptions/proofdraft,notcosttests.')
    (ROOT/'results/qwen_proof_matched_benchmark_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('all_passed','summary','communication_reduction_percent','primary_median_wall_reduction_percent','wall_ranges','stage_wall_upper_seconds','registered_files_verified')},indent=2))


if __name__=='__main__':main()
