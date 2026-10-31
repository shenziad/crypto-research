"""Exact coefficient-bound certificate for a registered future BFV response.

Proof obligations/assumptions are explicit in the paired note. Native tests are
correctness evidence, never used to infer a bound on unobserved ciphertexts.
"""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def describe(x):
    x=F(x)
    return dict(numerator=str(x.numerator),denominator=str(x.denominator),
                approximate=float(x),ceiling=math.ceil(x),ceiling_bits=math.ceil(x).bit_length())


def fast_CRT_sum(x,base):
    product=math.prod(base)
    return sum(((x%p)*pow(product//p,-1,p)%p)*(product//p) for p in base)


def small_RNS_identities():
    # Independently reconstruct integer CRT representatives; no SEAL calls.
    base=[3,5];q=15;M=16;Bbase=[17,19];B=math.prod(Bbase);msk=23
    lifted=0;floor_checks=0;SK_checks=0
    for x in range(q):
        S=fast_CRT_sum(M*x,base)
        r=(-S*pow(q,-1,M))%M
        if r>=M//2:r-=M
        assert (S+q*r)%M==0
        lift=(S+q*r)//M
        assert lift%q==x and F(-q,2)<=lift<F(q,2)+F(len(base)*q,M)
        lifted+=1
    for T in range(-2000,2001):
        S=fast_CRT_sum(T,base);assert 0<=S<len(base)*q and (T-S)%q==0
        Y=(T-S)//q
        assert -len(base)<F(Y)-F(T,q)<=0
        # Residues computed bydivide-and-floor agree with the exact integer.
        for p in Bbase+[msk]:assert ((T-S)*pow(q,-1,p))%p==Y%p
        SB=fast_CRT_sum(Y,Bbase)
        alpha=((SB-Y)*pow(B,-1,msk))%msk
        if alpha>msk//2:alpha-=msk
        assert (SB-alpha*B)%q==Y%q
        floor_checks+=1
    for Y in range(-9*B+1,9*B):
        SB=fast_CRT_sum(Y,Bbase)
        assert (SB-Y)%B==0
        actual_alpha=(SB-Y)//B;assert abs(actual_alpha)<F(msk,2)
        alpha=actual_alpha%msk
        if alpha>msk//2:alpha-=msk
        assert alpha==actual_alpha and (SB-alpha*B)%q==Y%q
        SK_checks+=1
    return dict(Montgomery_lifts=lifted,fast_floor_signed_inputs=floor_checks,SK_signed_inputs=SK_checks)


def main():
    out=ROOT/'results/bfv_response_bound_certificate.json';assert not out.exists()
    spec=json.loads((ROOT/'notes/20261031_bfv_response_proof_preregistration.json').read_bytes())
    native=json.loads((ROOT/'results/bfv_response_pipeline_check.json').read_bytes())
    assert native['all_passed'] and native['full_column_circuits']==6
    for row in native['results']:
        assert row['columns']==1024 and row['m_tilde']==2**32 and row['fresh_publickey_zero']
        assert row['base_B']==native['results'][0]['base_B'] and row['m_sk']==native['results'][0]['m_sk']
    p=json.loads((ROOT/'results/bfv_flooding_obligations.json').read_bytes())['profiles'][1]
    n=spec['polynomial_degree'];t=p['plain_modulus'];q=math.prod(p['first_moduli'])
    ell=len(p['first_moduli']);P=p['key_moduli'][-1];f=spec['noise_bits'];columns=spec['columns']
    baseB=native['results'][0]['base_B'];b=len(baseB);Bprod=math.prod(baseB)
    msk=native['results'][0]['m_sk'];M=native['results'][0]['m_tilde']
    # ClippedGaussian+int64cast: |error|<=19. Rounded(q*m/t)adds<=1/2.
    E=F(39,2);pmax=F(t-1,2);C=F(1,2)+F(ell,M)
    K=math.ceil(C*(n+1)+pmax/t+E/q)
    # ActualBEHZ: every lifted component <=Cq; middleproduct has2cross terms.
    Ymax=2*n*t*C*C*q+ell
    assert b+Ymax/Bprod<F(msk,2), 'SKsignedrecoverycapacitynotproved'
    assert ell<M/2
    basic=2*n*pmax*E + 2*t*n*K*E + F(t*n,q)*E*E + ell*(1+n+n*n)
    relin=F(n*19*sum(v-1 for v in p['first_moduli']),P)+F(n+1,2)
    response=columns*n*pmax*(basic+relin)
    # Conservative bound: ignore helpful1/Pscale inSEALpublic-zero keyswitch.
    public_zero=19*(2*n+1)+F(n+1,2)
    before_flood=response+public_zero
    # Convert rational(q/t)m center to integernearest center forsimulation.
    residual_cap=math.ceil(before_flood+F(1,2))
    alpha=3*(math.ceil(83/17)+math.ceil(163/17));assert alpha==45
    flooding=F(alpha*n*2*residual_cap,1<<f)
    assert flooding<F(1,1<<40)
    # Exact nearestcomponentdivision recurrence for size2/ternarysecret.
    error=before_flood+(1<<(f-1));levels=[];current=q
    assert error<F(q,2*t)
    for dropped in reversed(p['first_moduli'][1:]):
        current//=dropped;error=error/dropped+F(n+1,2)
        assert error<F(current,2*t)
        levels.append(dict(q=str(current),error_bound=describe(error),
                           correctness_margin=describe(F(current,2*t)-error)))
    assert current==int(p['last_q'])
    small=small_RNS_identities()
    hashes={name:sha(ROOT/name) for name in spec['source_sha256']}
    assert hashes==spec['source_sha256']
    files=['experiments/bfv_response_bound_certificate.py','experiments/bfv_response_pipeline_check.cpp',
        'results/bfv_response_pipeline_check.json','results/bfv_flooding_obligations.json',
        'notes/20261031_bfv_response_proof_preregistration.json','notes/20261031_bfv_response_proof_continuation.json']
    hashes.update({name:sha(ROOT/name) for name in files})
    result=dict(all_arithmetic_checks_passed=True,backend_integrated=False,full_system_certified=False,
        candidate_parameters=dict(n=n,t=t,first_q=str(q),special_prime=P,first_base_size=ell,
            ciphertexts_two_LUTs=alpha,columns=columns,position_slots=4096,noise_bits=f),
        bounds={name:describe(value) for name,value in dict(lift_ratio=C,phase_wrap_integer_cap=K,
            BEHZ_divided_component_abs=Ymax,SK_alpha_abs_cap=b+Ymax/Bprod,
            fresh_query_error=E,basic_multiplication_error=basic,relinearization_added_error=relin,
            response_error_before_rerandomization=response,public_zero_added_error=public_zero,
            before_flood_error=before_flood,integer_center_residual=residual_cap,
            joint45ciphertext_translated_flood_TV=flooding).items()},
        after_modswitch=levels,small_RNS_identity_checks=small,source_sha256=hashes,
        interpretation='Uniform coefficientbound for the registered future honestSEAL circuit shape, subjectto source-algebra correspondence provedinpairednote andstandardprimitivecorrectness. Fullciphertextsimulation additionallyrequires freshpublickeyzero andexplicitRLWE/PRNG assumption; oldFABLE hasnotbeenmodified/certified. Numericbound aloneisnotsecurityproof. No measurednoisemagnitudes used. No newprimitive claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_arithmetic_checks_passed=True,residual_cap=residual_cap,
        residual_bits=residual_cap.bit_length(),joint_TV=float(flooding),
        conservative_TV_bits=-math.log2(float(flooding)),small_RNS_identity_checks=small),indent=2))


if __name__=='__main__':main()
