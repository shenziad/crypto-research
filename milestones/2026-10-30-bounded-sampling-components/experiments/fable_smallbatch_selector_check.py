"""Public packed-selector check and explicit ideal-hash capacity calculation.

No attacks or data-dependent query deviations. The capacity statement assumes
independent uniform hashing; it is not a proof about std::hash or the full build.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def capacity(n,buckets,w=3,bits=54):
    with localcontext() as ctx:
        ctx.prec=80
        mu=Decimal(w*n)/Decimal(buckets)  # exact for power-of-two bucket count
        lnB=Decimal(buckets).ln().next_plus()
        ln2=Decimal(2).ln().next_plus()
        t=int(mu)+1
        while True:
            ctx.rounding=ROUND_FLOOR
            lower=(Decimal(t)/mu).ln().next_minus()
            product_lower=Decimal(t)*lower
            ctx.rounding=ROUND_CEILING
            upper=lnB-product_lower+Decimal(t)-mu
            residual=upper+Decimal(bits)*ln2
            if residual<0:break
            t+=1
        return dict(max_table_entries=n,buckets=buckets,replicas=w,
                    ideal_mean_load=str(mu),overflow_threshold=t,
                    load_capacity=t-1,log_union_bound_upper=str(upper),
                    log_budget_margin_upper=str(residual),failure_budget=f'2^-{bits}',
                    assumption='Ideal independent uniform hash placements; no standalone concrete std::hash/OPRF/cuckoo insertion failure theorem.')


def main():
    result=ROOT/'results/fable_smallbatch_selector_check.json';assert not result.exists()
    checked=0
    for m in (3,5,8,12,16):
        codes=[sum(1<<bit for bit in pair) for pair in itertools.combinations(range(m),2)]
        assert len(codes)==len(set(codes))
        for slots in (1,2,4,16):
            for selected_slot in range(slots):
                for selected_code in codes:
                    selected=[]
                    for slot in range(slots):
                        query=selected_code if slot==selected_slot else 0
                        for column,code in enumerate(codes):
                            value=int((query & code)==code)
                            if value:selected.append((slot,column))
                            checked+=1
                    assert selected==[(selected_slot,codes.index(selected_code))]
    cap=capacity(65536+255,512)
    out=dict(all_passed=True,public_selector_cells=checked,
             lemma='Each query code bit is zero outside its chosen subslot. Every column code has the same positive Hamming weight. Its product selector is one iff both the subslot and code equal the chosen ones. Thus a bucket contributes at most one data item; all other subslots have zero plaintext selectors.',
             capacity=cap,limits='Public selector law only, not actual FHE, full simulation, malicious queries or a complete failure-budget proof. Tests of actual unselected decrypted slots are still required.')
    result.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
