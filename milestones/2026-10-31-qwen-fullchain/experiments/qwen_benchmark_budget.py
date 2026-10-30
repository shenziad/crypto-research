"""Public parent-side execution accounting; never reads protocol secrets."""
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def reserve(label,pairs):
    spec=json.loads((ROOT/'notes/20261031_qwen_benchmark_budget.json').read_bytes())
    start=datetime.fromisoformat(spec['created_utc'])
    remaining=spec['maximum_wall_seconds']-(datetime.now(timezone.utc)-start).total_seconds()
    assert remaining>0,'fullchain stage wall budget exhausted'
    with (ROOT/'results/qwen_benchmark_ledger.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        ledger=json.load(f)
        assert ledger['native_pairs_reserved']+pairs<=spec['maximum_native_two_party_pairs']
        ledger['native_pairs_reserved']+=pairs
        ledger['events'].append(dict(label=label,pairs=pairs,reserved_utc=datetime.now(timezone.utc).isoformat()))
        f.seek(0);f.truncate();json.dump(ledger,f,indent=2);f.write('\n');f.flush()
    return remaining
