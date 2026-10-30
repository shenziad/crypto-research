"""Conditional exact arithmetic, NOT a bound on evaluated BFV circuit noise.

Reads public SEAL parameter contexts; does not generate ciphertexts, keys or
perform leakage/attack experiments. Complements source-level proof obligations.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT/'results/bfv_flooding_obligations.json'
    assert not out.exists()
    exe = ROOT/'toolchain/bfv-profile-audit/build/bfv_profile_audit'
    profiles = json.loads(subprocess.check_output([str(exe)], text=True, timeout=30))
    source = (ROOT/'third_party/FABLE-qwen18/extern/BatchPIR/header/utils.h').read_text()
    assert 'vector<int>{45, 45, 50}' in source and 'noise_bits = 69;' in source
    assert 'vector<int>{46, 40, 40, 46, 46}' in source and 'noise_bits = 149;' in source
    for p in profiles:
        n, t, f = p['n'], p['plain_modulus'], p['noise_bits']
        q = math.prod(p['first_moduli']); last_q = math.prod(p['last_moduli'])
        # Each response appends a19-bit index to the64/144-bit packed payload.
        width = p['plain_bits'] - 1
        alpha = 3 * (math.ceil(83/width) + math.ceil(163/width))
        p.update(key_q=str(math.prod(p['key_moduli'])), first_q=str(q), last_q=str(last_q),
                 first_q_bits=q.bit_length(), key_q_bits=math.prod(p['key_moduli']).bit_length(),
                 joint_two_LUT_response_ciphertexts=alpha,
                 flood_interval=f'[-2^{f-1},2^{f-1}-1]', flood_width=str(1<<f))
        # If each center has coefficient residual <=B, two centers differ by<=2B.
        # Independent uniform translated boxes: TV<=alpha*n*2B/2^f.
        b40 = (1 << f) // (2 * alpha * n * (1 << 40))
        p['conditional_40bit_translated_noise_residual_cap'] = str(b40)
        p['conditional_40bit_translated_noise_cap_bits'] = b40.bit_length()
        assert Fraction(alpha*n*2*b40,1<<f) <= Fraction(1,1<<40)
        assert Fraction(alpha*n*2*(b40+1),1<<f) > Fraction(1,1<<40)
        # Rounded(q*m/t) center contributes<=1/2. Sufficient pre-switch condition.
        F = 1 << (f-1)
        correct_cap = (q-t-1)//(2*t)-F
        p['conditional_pre_switch_correctness_residual_cap'] = str(correct_cap)
        assert 2*t*(correct_cap+F)+t < q
        assert not (2*t*(correct_cap+1+F)+t < q)
        # For size2 ct and ternary s, each nearest-integer component division
        # contributes <=(n+1)/2 raw residual. Bound each ofk steps unscaled.
        # This remains a conditional algebraic check; implementation rounding
        # correspondence and global evaluated residual have not been certified.
        steps = len(p['first_moduli'])-len(p['last_moduli'])
        post_bound = Fraction(last_q,q)*(b40+F+Fraction(1,2)) + Fraction(steps*(n+1),2)
        margin = Fraction(last_q,2*t)-post_bound
        p['conditional_modswitch_algebra_positive_margin_at_B40'] = margin > 0
        p['modswitch_steps'] = steps
        p['all_dimensions_source_checked'] = True
    paths = ['experiments/audit_bfv_flooding_obligations.py','experiments/bfv_profile_parameters.cpp',
             'toolchain/bfv-profile-audit/build/bfv_profile_audit',
             'third_party/FABLE-qwen18/extern/BatchPIR/header/utils.h',
             'third_party/FABLE-qwen18/extern/BatchPIR/src/utils.cpp',
             'third_party/FABLE-qwen18/extern/BatchPIR/src/batchpirserver.cpp',
             'third_party/FABLE-qwen18/extern/BatchPIR/src/batchpirclient.cpp',
             'third_party/FABLE/extern/SEAL/build/native/src/seal/util/config.h',
             'third_party/FABLE/extern/SEAL/native/src/seal/util/rlwe.cpp',
             'third_party/FABLE/extern/SEAL/native/src/seal/util/scalingvariant.cpp',
             'third_party/FABLE/extern/SEAL/native/src/seal/evaluator.cpp',
             'third_party/FABLE/extern/SEAL/native/src/seal/util/rns.cpp']
    result = dict(all_conditional_arithmetic_checks_passed=True, full_backend_certified=False,
                  profiles=profiles, created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={name:sha(ROOT/name) for name in paths},
                  obligations=['Prove a uniform pre-flood response residual bound for all valid query/table/pad values, including SEAL BFV multiplication, RNS basis rounding and relinearization. No suchbound is supplied here.',
                               'Prove the whole returned ciphertext distribution is simulatable, not just a translated coefficient-noise box; account for all ciphertext components and fresh rerandomization ifrequired.',
                               'Verify actual modswitch implementation matches conditional nearest-rounding algebra; include coefficient degree, allciphertexts and allcalls in composition.',
                               'Separately account for encryption, AES/GC/OT computational assumptions, OPRF consistency, pads, failure semantics and local sidechannel model.'],
                  limits='SEAL tc128 acceptance concerns its configured parameter table only; not circuitprivacy certification. B40 values are conditional sufficient caps, not observed noise, not measured leakage, not a proof of insecurity, and not certified feasible circuit bounds. Upstreamcase20 is a comparison only, not adopted or certified. No old performance number is assigned to it.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(full_backend_certified=False,profiles=profiles),indent=2))


if __name__ == '__main__':
    main()
