"""Produce independent AES reference using the project's CPU cryptography env."""
import hashlib
import json
import cryptography
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
from fused_gated_lut import ROOT

rng=np.random.default_rng(20261029);n=1024
keys=rng.integers(0,256,(n,16),dtype=np.uint8);plain=rng.integers(0,256,(n,16),dtype=np.uint8)
keys[0]=np.arange(16,dtype=np.uint8);plain[0]=np.arange(16,dtype=np.uint8)*17
expected=[]
for key,block in zip(keys,plain):
    enc=Cipher(algorithms.AES(key.tobytes()),modes.ECB()).encryptor()
    expected.append(np.frombuffer(enc.update(block.tobytes())+enc.finalize(),dtype=np.uint8))
path=ROOT/'results/dcf_cuda_aes_reference.npz'
np.savez_compressed(path,keys=keys,plain=plain,expected=np.asarray(expected))
(ROOT/'results/dcf_cuda_aes_reference.json').write_text(json.dumps(dict(cases=n,cryptography_version=cryptography.__version__,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),purpose='Independent deterministic software AES reference; synthetic public test keys only.'),indent=2)+'\n')
print('saved independent AES reference',n,flush=True)
