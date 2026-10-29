"""Exhaustive integer signed-clamp lifting identity, independent of DCF."""
import json
from pathlib import Path


def main():
    ring=256;count=0
    for maximum in (1,3,31):
        for mask in range(ring):
            def coeff(h):
                z=(h-mask)%ring;s=z if z<ring//2 else z-ring
                return (1,s-h) if -maximum<=s<maximum else (0,min(max(s,-maximum),maximum))
            base=coeff(ring-1);terms=[]
            for threshold in ((mask+maximum)%ring,(mask+ring//2)%ring,(mask-maximum)%ring):
                left,right=coeff((threshold-1)%ring),coeff(threshold)
                terms.append((threshold,left[0]-right[0],left[1]-right[1]))
            for h in range(ring):
                slope,constant=base
                for threshold,dm,dc in terms:
                    if h<threshold:slope+=dm;constant+=dc
                z=(h-mask)%ring;s=z if z<ring//2 else z-ring
                assert slope*h+constant==min(max(s,-maximum),maximum)
                count+=1
    out=dict(small_input_bits=8,widened_integer_identity_pairs=count,all_passed=True,
        limits='Exhaustive clear integer identity; native DCF checked separately.')
    (Path(__file__).resolve().parents[1]/'results/gated_wide_clamp_identity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
