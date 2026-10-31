"""Public parent-side execution accounting; never reads protocol secrets."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify_ready():
    ready=json.loads((ROOT/'sources/fable_proof_backend_build/ready_manifest.json').read_bytes())
    for name,h in ready['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name


def reserve(label,pairs):
    verify_ready()
    spec=json.loads((ROOT/'notes/20261031_fable_proof_integration_preregistration.json').read_bytes())
    start=datetime.fromisoformat(spec['created_utc'])
    remaining=spec['maximum_wall_seconds']-(datetime.now(timezone.utc)-start).total_seconds()
    assert remaining>0,'fullchain stage wall budget exhausted'
    with (ROOT/'results/fable_proof_integration_ledger.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        ledger=json.load(f)
        assert not ledger.get('stage_closed',False),'stage alreadyclosed'
        assert ledger['native_pairs_reserved']+pairs<=spec['maximum_native_two_party_pairs']
        ledger['native_pairs_reserved']+=pairs
        ledger['events'].append(dict(label=label,pairs=pairs,reserved_utc=datetime.now(timezone.utc).isoformat()))
        f.seek(0);f.truncate();json.dump(ledger,f,indent=2);f.write('\n');f.flush()
    return remaining


def finish(label,success):
    with (ROOT/'results/fable_proof_integration_ledger.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);ledger=json.load(f)
        matches=[e for e in ledger['events'] if e['label']==label]
        assert len(matches)==1 and 'completed_utc' not in matches[0]
        matches[0].update(completed_utc=datetime.now(timezone.utc).isoformat(),success=bool(success))
        f.seek(0);f.truncate();json.dump(ledger,f,indent=2);f.write('\n');f.flush()
