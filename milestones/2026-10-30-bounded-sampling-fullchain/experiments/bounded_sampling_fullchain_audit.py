"""Read-only replay/hash/native-zero audit of all completed full-chain runs."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import run_bounded_sampling_pipeline as sequential
import run_bounded_sampling_batched as batched

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    destination=ROOT/'results/bounded_sampling_fullchain_audit.json';assert not destination.exists()
    inputs=0;transcripts=0;pairs=0;alias_rows=0;a_rows=0;zeros=0;proposals=0;max_load=0
    raw=[]
    for stem in ('bounded_sampling_pipeline','bounded_sampling_batched','bounded_sampling_fullchain_benchmark'):
        artifact=ROOT/'results'/f'{stem}.json'
        data=json.loads(artifact.read_bytes());assert data['all_passed']
        for name,digest in data['input_sha256'].items():assert sha(ROOT/name)==digest,name;inputs+=1
        for row in data['results']:
            for name,digest in row['audit_file_sha256'].items():assert sha(ROOT/name)==digest,name;transcripts+=1
            worker_metrics=[ROOT/n for n in row['audit_file_sha256'] if n.endswith('/worker_metrics.json')]
            assert len(worker_metrics)==2
            folders=sorted(p.parent for p in worker_metrics)
            folder=folders[0].parent
            client=(ROOT/'results/bounded_sampling_pipeline_runs/ptb/client_initial.npz') if stem=='bounded_sampling_fullchain_benchmark' else folder/'client_initial.npz'
            with np.load(client) as source:
                xi=source['x_integer']
            # Benchmark fixtures are pinned to the original PTB initial files.
            server=(ROOT/'results/bounded_sampling_pipeline_runs/ptb/server_initial.npz') if stem=='bounded_sampling_fullchain_benchmark' else folder/'server_initial.npz'
            with np.load(server) as source:ai=source['a_integer'];bi=source['b_integer']
            is_batch='alias_lookup' in row['parties'][0]
            replay=(batched if is_batch else sequential).audit(folder,xi,bi,ai)
            for field in ('all_correct','proposals_checked','integer_rows_checked','final_token_matches',
                          'total_wire_bytes','stage_wire_bytes','public_failure','audit_file_sha256'):
                assert replay[field]==row[field],(stem,folder.name,field)
            logs=[(folders[0]/'row_native.log',folders[1]/'row_native.log')]
            if is_batch:
                logs.append((folders[1]/'alias_row_native.log',folders[0]/'alias_row_native.log'))
                alias_rows+=256
            for clientlog,ownerlog in logs:
                matches=re.findall(r'SMALLBATCH_UNSELECTED_ZERO_CHECKS=(\d+)',clientlog.read_text())
                loads=re.findall(r'SMALLBATCH_MAX_LOAD=(\d+)',ownerlog.read_text())
                assert len(matches)==len(loads)==1 and int(matches[0])==345600
                assert int(loads[0])<=583
                zeros+=int(matches[0]);max_load=max(max_load,int(loads[0]))
            pairs+=1;proposals+=201;a_rows+=256
        raw.append(dict(path=str(artifact.relative_to(ROOT)),sha256=sha(artifact)))
    out=dict(all_passed=True,complete_chain_pairs=pairs,secret_proposals_verified=proposals,
        integer_A_rows_verified=a_rows,alias_rows_verified=alias_rows,
        unselected_plaintext_slots_checked_zero=zeros,max_observed_bucket_load=max_load,
        pinned_input_hash_checks=inputs,transcript_file_hash_checks=transcripts,inputs=raw,
        source_sha256=sha(Path(__file__)),
        limits='Replays stored full-chain functional audits and checks allnativezero assertions and rawfile hashes. Does not prove implementation security, concretehash/cuckoofailure, freshquality, optimizeddensebaseline superiority or independentnovelty.')
    destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
