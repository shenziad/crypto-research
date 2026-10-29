"""Exact finite shift identity and real two-party full32-domain checks."""
import hashlib
import json
import numpy as np
from gated_lut_strong_baselines import run,ROOT
from dpf_aes_native import LIB


def main():
    xs=np.arange(256,dtype=np.int64);signed=(xs^128)-128
    pairs=0
    for r in range(256):
        h=(xs+r)&255;theta=(r+128)&255
        out=(h>>3)-(r>>3)-(h%8<r%8).astype(np.int64)+32*((h<theta).astype(np.int64)-int(r<128))
        assert np.array_equal(out,signed//8);pairs+=len(xs)
    rng=np.random.default_rng(291901)
    boundaries=np.array([-(1<<31),-(1<<31)+1,-65537,-65536,-257,-256,-255,-1,0,1,255,256,257,65535,(1<<31)-2,(1<<31)-1],dtype=np.int64)
    values=np.concatenate([boundaries,rng.integers(-(1<<31),1<<31,2048,dtype=np.int64)])
    a=rng.integers(0,256,len(values),dtype=np.uint8);table=np.ones(256,dtype=np.uint64)
    checks=[run(a,values.view(np.uint64),m,table_values=table,b_value_bits=32) for m in ('beaver32','fused32')]
    out={'exhaustive_8bit_secret_mask_pairs':pairs,'shift_removed_bits':3,'all_integer_identities_passed':True,
         'full32_domain_mpc_checks':checks,'native_sha256':hashlib.sha256(LIB.read_bytes()).hexdigest(),
         'limits':'Tests exact signed shift and32->64 output function on all small-ring secret/mask pairs and wider boundary/random values. Standard FSS/DCF construction assumptions and implementation audit remain separate.'}
    (ROOT/'results/faithful_shift_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
