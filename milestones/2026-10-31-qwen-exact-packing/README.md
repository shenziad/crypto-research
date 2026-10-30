# Exact private-field packing and full three-path Qwen comparison

Same frozen model/F16/F20/PWL/E31/p44/K201 and client-only token output. Alias uses fixed64 bits, selected rank8 Arow uses144 bits instead of uniform256; this is exact representation with the backend's existing width support, not quantization or a new primitive.

Four preset mixed-order blocks all pass. Median bytes / whole-pair wall:

|Complete path|Bytes|Wall ms|
|---|---:|---:|
|Optimized denseCDF95|3,504,947,076|24,071.87|
|Wide256 private rejection|1,806,419,191.5|11,761.63|
|Alias64/Arow144 private rejection|976,779,523.5|8,408.28|

Packed vs dense:72.13% less communication/65.07% less wall; packed vs wide:45.93%/28.51%. Four observations perpath/one performance context, fixed orders not perfectly position-balanced; not globalSOTA/WAN/statistical significance or full-private-backbone result. Allsetup, local codec, LUT and arithmetic costs included; public backbone/input-fixture preparation excluded and asymmetric local prep is documented.

See [outcome](notes/20261031_qwen_exactpacking_outcome.md), [threeway summary](results/qwen_packed_threeway_benchmark.json), [audit](results/qwen_packed_threeway_audit.json), and [codec checks](results/qwen_exact_row_codec_check.json). Two-corpus full conformance,512 privateLUT rows,201 alias boundary cases and2,097,152 codec coefficient patterns pass. Stage uses47native pairs/665.18s within50pairs/3600s cap.731 registered-file hashes/replay checks pass; no model or parameter retuning.

## Separate analytical security candidate

A future modified backend using domain-separated tags, balanced distinct-triple maps, full augmenting matching and4096 position slots has an exact Hall-witness statistical candidate bound below2^-55 for twoLUT calls under stated ideal/PRP assumptions. See [construction and missing work](notes/20261031_cuckoo_certification_candidate.md) and [certificate](results/cuckoo_hall_candidate_certificate.json).

That candidate is not implemented and does NOT certify the currentstd::hash/500-step backend or its measured performance. BFV correctness/noise hiding and whole shared-output composition remain open. The loose legacy upperbound is neither an observed failure rate nor a lowerbound. No attacks/failure search were performed; no new Hall theorem/hash primitive/CCF-level claim.

## Reproduction boundary

Own source, registrations and layered build recipes are copied verbatim. Result JSONs are explicitly reduced summaries without private per-file hashes/logs. Full original records, shares/coins/tokens/captures/binaries and checksum checkpoint remain on NEWSMY; model/data/env/third-party source dependencies follow the earlier pinned milestones. This is not a standalone clean-clone installer. Existing frozen quality signals and original50% gate result are preserved; positive meaningful signals continue under the documented user decision policy.
