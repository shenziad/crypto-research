"""Account for cached duplicate proposals in the frozen sampling screen."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'results'
    source = folder/'fixed_proposal_sampling_screen.json'
    doc = json.loads(source.read_text())
    archive = folder/'fixed_proposal_sampling_inputs.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == doc['input_archive_sha256']
    result = {}
    with np.load(archive) as arrays:
        for name in ('uniform', 'calibration_mean', 'calibration_minimax'):
            q = arrays['proposal_'+name]
            bound = max(max(r['oracle_context_envelopes']) for r in doc['rows'] if r['proposal'] == name)
            count = math.ceil(math.log(2**-40)/math.log1p(-1/bound))
            distinct = float((-np.expm1(count*np.log1p(-q))).sum())
            result[name] = dict(common_envelope_on_observed_heldouts=bound, K=count,
                expected_unique_candidates=distinct, fraction_vocabulary=distinct/doc['vocabulary'])
    payload = dict(analysis=result, parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formula='E distinct candidates=sum_i[1-(1-q_i)^K] for independent proposals; finite observed-set envelope only. Deduplicating head projections does not remove secret sampling/selection costs; those are not implemented.')
    path = folder/'fixed_proposal_sampling_deduplication.json'
    if path.exists():
        assert json.loads(path.read_text()) == payload
    else:
        path.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
