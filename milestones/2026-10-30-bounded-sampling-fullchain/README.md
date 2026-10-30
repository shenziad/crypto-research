# Complete bounded private-head sampling prototype

2026-10-30. The restricted finite private-head chain is now connected; the research goal is still active. No complete private backbone, independent new primitive, optimized dense-baseline advantage or CCF C+ claim.

Read [the full outcome and cost limits](notes/20261030_bounded_sampling_fullchain_outcome.md), [the conditional hybrid proof draft](notes/20261030_bounded_sampling_hybrid_proof.md), and [the next strong-baseline contract](notes/20261030_dense_head_baseline_contract.md). Earlier quality/range/Bernoulli/FABLE components and pins remain in [the preceding milestone](../2026-10-30-bounded-sampling-components/README.md).

Both implementations execute actual secret alias proposals -> private integerA row lookup -> private Bx/Arow/floor/tanh -> secret Bernoulli acceptance/first-success -> client-only final token/failure. Each logical party passes only its own shares; the parent reads intermediates only after both workers/nativechildren terminate. Independent replay checks12complete runs,2412secret proposals,3072integerA rows,1536alias rows and6220800unselected HE slots equalzero.

On the same frozen PTB input,4alternating paired runs perimplementation give medians:

| Backend choice | Total application bytes | Max worker elapsed ms |
|---|---:|---:|
| Sequential existing PythonOT/Yao alias | 620989473.5 | 45072.461 |
| Batched existing FABLE/SCI alias | 821985397.5 | 3839.315 |

This is a local latency/bandwidth tradeoff between two research implementations, not an algorithmicSOTA speedup. Wholechain counting includes setup/keys/table/mask/output; publicbackbone and initialcaptured x/q are excluded. Two reused model contexts are functional checks, not new blind quality or continuous generation evidence.

Primary audit confirms TCC2022Protocol4 already has secretproposal/corrective rejection/fixedbatch firstsuccess; that skeleton is notnew. Remaining potential value would require LLM architecture/quality/cost benefits against a strong dense private-head baseline, which has not yet been implemented.

## Reproduction scope

These are exact local research sources/results and preregistrations, not a fresh-clone installer. Existing pinned SCI/FABLE libraries, prior component patches/environment, own adapter NPZ/captured x/q and the local directory layout are required or need deliberate adaptation. Official model/corpus and all local input hashes are recorded in the preceding milestone and current preregistrations. Raw logical-party input/share/output files, own binaries, environments, model/corpus assets and primary third-party HTML/PDF stay on NEWSMY. Public records include hashes and the after-exit auditing code; hashes alone do not demonstrate cross-machine reproduction. The parent audit files are logs/inputs/outputs/metrics, not a packet capture of every wire message.

The finite samplingdistance2.6430181366e-13 applies only to fixedfiniteq/delta; it excludes modelapproximation, concretehash/cuckoo failures and implementationsecurity. Hybrid proof assumes ideal static-passive components; actual joint reduction, malicious/adaptive/UC security and optimizeddense fullsampling comparison remain open.
