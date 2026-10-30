"""Analytical candidate for a future certified private-LUT backend.

This does NOT certify the current std::hash/500-step FABLE implementation.
Uses exact Hall-witness sums under independent uniform distinct3 choices;
capacity bound reuses the recorded directed-Decimal Chernoff certificate.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

from fable_smallbatch_selector_check import capacity

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def hall_sum(n,m):
    denominator=math.comb(m,3)
    total=F(0)
    for s in range(4,n+1):
        total+=math.comb(n,s)*math.comb(m,s-1)*F(math.comb(s-1,3),denominator)**s
    return total


def describe(value):
    with localcontext() as c:
        c.prec=60;c.rounding=ROUND_CEILING
        upper=Decimal(value.numerator)/Decimal(value.denominator)
    k=value.denominator.bit_length()-value.numerator.bit_length()
    if k>=0:
        while value.numerator*(1<<k)>=value.denominator:k-=1
        while value.numerator*(1<<(k+1))<value.denominator:k+=1
    def digest(n):return hashlib.sha256(n.to_bytes((n.bit_length()+7)//8,'big')).hexdigest()
    return dict(decimal_upper=str(upper),strict_upper_power2_negative=k,
        numerator_sha256=digest(value.numerator),denominator_sha256=digest(value.denominator),
        numerator_bits=value.numerator.bit_length(),denominator_bits=value.denominator.bit_length())


def main():
    outpath=ROOT/'results/cuckoo_hall_candidate_certificate.json';assert not outpath.exists()
    N=262144+255;B=2048;loadcap=600;queries=256
    cap=capacity(N,B,bits=60);assert cap['load_capacity']<=loadcap
    query_failure=hall_sum(queries,B)
    legacy_feasibility_bound=B*hall_sum(loadcap,960)
    candidate_feasibility_bound=B*hall_sum(loadcap,4096)
    # Future construction: two domain-separated128-bit AES tags per unique
    # row id; balanced mixed-radix maps to ordered distinct bucket/position
    # triples. Full augmenting-path matching removes the500-step walk event.
    DB=B*(B-1)*(B-2);M=4096;DP=M*(M-1)*(M-2)
    mapping_bias=F(N*(DB+DP),1<<128)
    Q=2*N;prp_to_prf=F(Q*(Q-1),1<<129)
    per_lookup=F(1,1<<60)+query_failure+candidate_feasibility_bound+mapping_bias+prp_to_prf
    two_lookups=2*per_lookup
    assert two_lookups<F(1,1<<54)
    result=dict(all_analytical_checks_passed=True,current_backend_certified=False,
        candidate_parameters=dict(unique_ids_max=N,buckets=B,load_cap=loadcap,
            query_count=queries,choices=3,position_slots=4096,AES_domains=2),
        capacity_directed_certificate=cap,query_Hall_bound=describe(query_failure),
        legacy960_worstload_Hall_upper=describe(legacy_feasibility_bound),
        candidate4096_worstload_Hall_upper=describe(candidate_feasibility_bound),
        mapping_TV_bound=describe(mapping_bias),PRP_switching_bound=describe(prp_to_prf),
        candidate_per_lookup_statistical_bound=describe(per_lookup),
        candidate_two_lookup_statistical_bound=describe(two_lookups),
        formula='Hall(n,m)=sum(s=4..n) C(n,s) C(m,s-1) (C(s-1,3)/C(m,3))^s. Serverunion<=B*Hall(loadcap,M) conditionaloncapand independent bucket/position tags.',
        mandatory_changes=['Independent domain-separated AES tags and explicit balanced triple maps; currentstdhashdoesnotqualify.',
            'Complete augmenting-path matching on clientandserver; current500-step randomwalknotcovered.',
            'Publicpositionslots4096 and actualPIR layout, BFV correctness/noiseflood analysis beforeclaimingfullbackendsecurity.',
            'Verifydedup producesdistinctvalidids independentoffreshkey; retainfreshshared-outputpadsandalljointcomposition obligations.'],
        source_sha256=sha(Path(__file__)),
        capacity_source_sha256=sha(ROOT/'experiments/fable_smallbatch_selector_check.py'),
        limits='Pureconstruction/correctnessprobabilityanalysis, no attacks orfailuresearch. A larger worstload upperbound forlegacy960 is not a lowerbound or observed failure rate. Candidate isnotimplemented/certified/currentperformance result. Requiresideal independent tags plus AESPRP advantage; totals excludeBFV/garbling/OT/composition errors. Uniformmaps areconditionedonvaliduniqueinputids/freshkey; no adversarial adaptivekeyclaim. No newhashing theorem/primitive/CCFclaim.')
    outpath.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_analytical_checks_passed','current_backend_certified',
        'query_Hall_bound','legacy960_worstload_Hall_upper','candidate4096_worstload_Hall_upper',
        'candidate_two_lookup_statistical_bound')},indent=2))


if __name__=='__main__':main()
