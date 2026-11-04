# Phase17: complete full-vocabulary fixed-K-failure dense control

Six private-head correctness controls and eight full-vocabulary protocol pairs passed after-exit audits. This integrates the same finite token-or-FAIL law as the K=244 candidate into a complete dense private head; it is not the earlier 2.50 MB precomputed-weight core.

| Input | Candidate bytes (2 runs) | Dense bytes (2 runs) | Mean candidate reduction |
|---|---:|---:|---:|
| train_first | 1,539,064,399, 1,539,068,117 | 3,508,012,338, 3,508,012,338 | 56.1271% |
| point_zero | 1,539,067,216, 1,539,063,369 | 3,508,012,338, 3,508,012,338 | 56.1271% |

Vocabulary 151,936; hidden width 896; rank 8; the same finite alias44 masses totaling 2^62, fixed16/PWL/E31 thresholds, K and FAIL identity. All native traffic in both directions and fresh native setup are charged. Per-party fixture read, encoding and proposal compilation are measured. Common cached model quantization/public prefill are reported separately. The input is one previously fixed training-context prefill plus a zero-hidden/point-EOS calibration, with no quality filtering.

The dense control privately tests K hidden 93-bit uniforms against the exact normalizer and performs a weighted CDF scan at the first accepted uniform; only client token-or-FAIL is released. Controlled correctness runs cover all-reject, late acceptance, zero-mass entries and the full-normalizer endpoint. All reconstructed head intermediates, E31 values, weights, prefixes, first-success flag, CDF comparisons and outputs match the integer reference; the server output files are empty.

The equality is of the ideal finite output distribution including FAIL. The actual GC/OT composition and the candidate HE/RLWE known-key circuit-privacy assumptions remain open proof obligations. Native correctness is not a complete joint security proof.

Timing uses only two runs per input and protocol on one loopback machine, so it is descriptive. This is one-call full private-head execution, not complete autoregressive generation or a WAN result. Full intermediate party records, raw inputs and source are not included here.

The prior semantic gate remains failed under non-independent assistant review. No new semantic generation, retraining, reviewer adjustment or independent-test evaluation occurred. Both independent test commitments remain sealed. MoE and C10 remain paused.

EDEN full construction remains unreviewed because public full text was inaccessible; metadata alone cannot establish novelty. Strong baseline validation improves the evidence but does not establish a publication-level result.
