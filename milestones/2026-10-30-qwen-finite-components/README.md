# Qwen fixed finite-quality and native selected-head components

SameF16/F20 implementation preserves frozen quality signals across8192 reused holdout predictions: finite NLL gains0.12240328/PTB and0.00649254/Wiki, versus floating0.12210977/0.00626282. This is implementation conformance, not another fresh quality study or precision optimization.

Ten real2party Qwen896/index18 arithmetic components pass, including+/-52 coordinates, token151935, equality rejection and first/last/all-failure controls. All1010 selected rows and t/z/PWL/E31 are verified after both processes exit. The201-row component uses7,799,411 bytes; it excludes18-bit private LUT/alias and fullsampler. No complete Qwen speed claim.

The registered stage capped8192 positions,10 native pairs and1800seconds; observed completion upper87.25seconds. See [outcome](notes/20261030_qwen_finite_stage_outcome.md), [quality results](results/qwen_fixed16_quality.json), [native summary](results/qwen_selected_exp31.json), and [user decision rule](RESEARCH_SCOPE.md). Positive meaningful signals cannot be archived solely because a preset metric was missed: distinguish unmet target/core falsification/new phenomenon and permit one explicit-cost targeted supplement. Original gate records stay unchanged.

Own sources/preregistrations are copied verbatim. Native/audit JSONs are explicitly reduced summaries without per-file private hashes/logs. Original frozen records/shares/coins/weights/binaries remain on NEWSMY. Requires the prior pinned SCI/FABLE dependencies, source helpers and local model/data fixtures; not a clean-clone installer. Complete18-bit lookup/CDF95 paths, strong whole-sampler benchmark, concrete failure and joint security, and independent CCF-level contribution remain unfinished.
