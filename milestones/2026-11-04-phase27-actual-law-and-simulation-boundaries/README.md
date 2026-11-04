# Phase 27 — exact error-law interface and conditional simulation boundaries

The selected repaired backend remains **phase26-cot-20261104**. No protocol, archive, parameter or worker changed. This stage runs only fixed public probes and small finite-model checks, with no GPU, full sampler rerun, training, task/seed sweep, installation or sealed-test access.

The actual linked normal sampler was replayed on1048576 coefficients and all4194304 RNS limbs. Source conversion truncates clipped floating normals toward zero, so it must not be labeled a standard discrete Gaussian with sigma3.2. Observed sd is2.82834; a continuous clipped-then-truncate reference gives2.82830. These are implementation/analytic observations, not an exact binary64 PMF, coefficient-independence proof or hardness estimate. The assumption must specify the pinned finite-word/vector sampler, including polar cache and rounding; the previous support bound and failure ledger are retained.

A proposed conditional half-gate simulation argument retains publicly generated active labels instead of replacing a public-seed PRG output by uniform. It requires an independent domain-separated random oracle, ideal chosen COT, fresh hidden offset and distinct tweaks. An exact tiny one-gate enumeration checks its algebra. This is a proof sketch for independent review, not closure of the current AES/IKNP/common-model implementation. See the formal-model context in the [CRYPTO2020 half-gates paper](https://link.springer.com/content/pdf/10.1007/978-3-030-56880-1_28.pdf).

A correct-decryption finite ciphertext example demonstrates why marginal phase masking is insufficient: correlated masking gives full-cipher TV1 despite phase marginal TV1/64; independent flooding gives full-cipher TV1/64. Current source uses a fresh flood stream in both active variants, supporting the intended ideal premise. Actual PRNG/auxiliary-view independence and subsequent GC/full-record consistency still need reduction; this is not an attack on the current implementation.

Even a hypothetical seed-preserving ROM/direct-NP budget retains about4.48 million curve multiplications and conditional1.660GB full candidate bytes (~7.87% above its reference). It is not implemented or end-to-end validated. Earlier direct-NP scaling negatives remain; no free speedup or full protocol theorem is claimed.

Specific circuit-privacy theorems in [CRYPTO2016](https://link.springer.com/content/pdf/10.1007/978-3-662-53008-5_3.pdf) concern GSW/gadget constructions, not a drop-in BFV proof. [TinyLabels](https://eprint.iacr.org/2024/2048) is an input-compression lead with metadata/abstract checked; its full PDF returned403, so theorem/source compatibility is unverified. Restricted accesses were not retried or bypassed.

Private-adapter application value is still unestablished. Small-menu policy decisions require strong small-classifier/ranker baselines; open-ended private-preference generation still lacks a validated matched adapter and independent semantic evidence. Earlier failures remain negative. This does not prove no application exists, and does not justify selecting favorable prompts or opening sealed tests.

No security bits or publication outcome are asserted. MoE/C10 remain paused. Only aggregate information is published; detailed source, proof notes, patches, party records and weights remain local pending explicit authorization.
