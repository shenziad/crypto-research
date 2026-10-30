"""Retrospective saved-file replay of complete balanced Qwen samplers."""
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
from run_qwen_sampling_exp31 import audit as wide_audit
from run_qwen_packed_sampling import audit as packed_audit

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    # An audit invocation is not permitted while one of these native binaries
    # is still active. Check executable argv; shell command strings do not count.
    active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if argv and Path(argv[0].decode(errors='replace')).name in {
            'qwen_dense_cdf31','qwen_selected_exp31','qwen_alias_batch','private_row_harness'}:
            active.append(str(p.parent))
    assert not active,active
    regp=ROOT/'notes/20261031_qwen_packed_threeway_benchmark_preregistration.json'
    reg=json.loads(regp.read_bytes())
    rp=ROOT/'results/qwen_packed_threeway_benchmark.json'
    result=json.loads(rp.read_bytes());assert result['all_passed']
    assert result['preregistration_sha256']==sha(regp)
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
    for stem,regname in [('qwen_exactpacking_controls','qwen_exactpacking_controls'),
                          ('qwen_packed_sampling','qwen_packed_sampling')]:
        rr=json.loads((ROOT/f'results/{stem}.json').read_bytes())
        preg=ROOT/f'notes/20261031_{regname}_preregistration.json'
        assert rr['all_passed'] and rr['preregistration_sha256']==sha(preg)
        check(rr);check(json.loads(preg.read_bytes()))
    # Also re-execute the saved packed conformance audit for both corpora.
    for corpus in ('ptb_test','wiki_test'):
        folder=ROOT/'results/qwen_packed_sampling_runs'/corpus
        with np.load(folder/'client_initial.npz') as ar:xx=ar['x_integer'];qq=ar['q']
        with np.load(folder/'server_initial.npz') as ar:aa=ar['a_integer'];bb=ar['b_integer']
        packed_audit(folder,xx,bb,aa)
        mass0=build(qq,44)[2]
        with np.load(folder/'party0/client_alias_audit.npz') as ar:assert np.array_equal(ar['mass'],mass0)
    common=ROOT/'results/qwen_packed_threeway_benchmark_runs/common'
    with np.load(common/'client.npz') as archive:x=archive['x_integer'];q=archive['q']
    with np.load(common/'server.npz') as archive:a=archive['a_integer'];b=archive['b_integer']
    mass=build(q,44)[2][:151936].astype(np.uint64)
    assert sum(map(int,mass))==1<<62
    full=expected(x,b,a)
    assert hashlib.sha256(mass.astype('<u8').tobytes()).hexdigest()==result['target_q_mass_sha256']
    assert hashlib.sha256(full[-1].astype('<u8').tobytes()).hexdigest()==result['target_E31_vector_sha256']
    seen=[]
    for row in result['results']:
        folder=common.parent/f"{row['repetition']}_{row['variant']}"
        if row['variant']!='dense':
            fresh=(packed_audit if row['variant']=='packed' else wide_audit)(folder,x,b,a)
            assert fresh['total_wire_bytes']==row['total_wire_bytes']
            with np.load(folder/'party0/client_alias_audit.npz') as archive:
                assert np.array_equal(archive['mass'][:151936],mass)
        else:
            paths=[folder/f'party{i}' for i in (1,2)]
            for name,width,value in zip(
                ('raw_t_add41.bin','t_add26.bin','raw_z_add44.bin','delta_add44.bin','exp_add33.bin'),
                (41,26,44,44,33),full):
                for p in paths:assert (p/name).stat().st_size==8*value.size
                assert np.array_equal(reconstruct(paths,name,width),value),name
            cdf_audit(folder,mass.tolist(),full[-1].tolist())
            parties=[json.loads((p/'metrics.json').read_bytes()) for p in paths]
            for c in range(3):
                assert parties[0]['end_counters'][c]==parties[1]['end_counters'][c+3]
                assert parties[1]['end_counters'][c]==parties[0]['end_counters'][c+3]
            assert sum(sum(p['end_counters'][:3]) for p in parties)==row['total_wire_bytes']
        seen.append(row['variant'])
    assert seen==[v for order in reg['orders'] for v in order]
    summary={}
    for variant in ('dense','wide','packed'):
        rows=[r for r in result['results'] if r['variant']==variant];assert len(rows)==4
        summary[variant]=dict(observations=len(rows),
            median_total_wire_bytes=statistics.median(r['total_wire_bytes'] for r in rows),
            median_whole_pair_wall_ms=statistics.median(r['whole_pair_wall_ms'] for r in rows),
            median_participant_elapsed_ms=statistics.median(r['participant_elapsed_ms'] for r in rows))
    assert summary==result['summary']
    reduction={}
    for target,baseline in [('packed','wide'),('packed','dense'),('wide','dense')]:
        reduction[target+'_vs_'+baseline]={key:100*(1-summary[target][key]/summary[baseline][key])
            for key in ('median_total_wire_bytes','median_whole_pair_wall_ms')}
    budget=json.loads((ROOT/'notes/20261031_qwen_exactpacking_budget.json').read_bytes())
    lp=ROOT/'results/qwen_exactpacking_ledger.json';ledger=json.loads(lp.read_bytes())
    assert ledger['native_pairs_reserved']==47<=budget['maximum_native_two_party_pairs']
    elapsed=rp.stat().st_mtime-datetime.fromisoformat(budget['created_utc']).timestamp()
    assert elapsed<budget['maximum_wall_seconds']
    ledger.update(stage_closed=True,actual_native_pairs=47,completion_wall_upper_seconds=elapsed)
    lp.write_text(json.dumps(ledger,indent=2)+'\n')
    out=dict(all_passed=True,retrospective=True,created_utc=datetime.now(timezone.utc).isoformat(),
        summary=summary,reduction_percent=reduction,registered_files_verified=len(verified),
        complete_benchmark_runs_replayed=12,benchmark_native_pairs=36,
        conformance_native_pairs=11,benchmark_stage_wall_upper_seconds=elapsed,
        benchmark_stage_max_wall_seconds=budget['maximum_wall_seconds'],
        result_sha256=sha(rp),preregistration_sha256=sha(regp),source_sha256=sha(Path(__file__)),
        verified_files=verified,
        limits='After-exit reconstruction and file/counter consistency; not independent implementation or security proof. Four observations perpath/onePTBcontext, fixed3wayorders notperfectpositionbalanced, no uncertainty interval, WAN orglobalSOTA claim. Exactpackingnofunction/modelchange; existingbackendwidthsupportnotnewprimitive. Separate model/alias/crypto failures remain. Conformance has2corpora, benchmark reusesone. Participant timings diagnostic; primarywhole-pair includesasymmetric workerlocalprep describedinregistration.')
    destination=ROOT/'results/qwen_packed_threeway_audit.json';assert not destination.exists()
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='verified_files'},indent=2))


if __name__=='__main__':main()
