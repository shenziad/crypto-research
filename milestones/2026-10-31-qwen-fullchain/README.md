# Qwen complete bounded private-head sampling

Both complete151936-token paths now execute with client-only token output: optimized denseF16/PWL/E31/CDF95 and secretFABLE18 alias/selected-row/E31 rejection. Same frozen896-dimensional input, rank8 adapter and integer probability law.

Four alternating paired comparisons all pass. Median whole-pair wall: dense24,572.31ms vs rejection12,061.63ms; wire3,504,947,076B vs1,806,418,740.5B. Observed reductions48.46% bytes/50.91% wall. These are pinned local research implementations, not globalSOTA or complete private-backbone inference.

See [outcome/limits](notes/20261031_qwen_fullchain_outcome.md), [finite functionality and conditional composition](notes/20261031_qwen_finite_functionality.md), [benchmark summary](results/qwen_matched_exp31_benchmark.json), [audit summary](results/qwen_fullchain_benchmark_audit.json), and [prior update](notes/20261031_bounded_head_prior_update.md).

The prior frozen quality signal is preserved:PTB0.1221nat/Wiki0.00626nat NLL improvements; same finite arithmetic retains both on8192 reused positions. Original50% validation gate remains missed. Positive observations justify continuing contribution/security work under the user's documented decision policy, not archiving solely on that gate or claiming CCF success.

## Budgets and exact law

Conformance stage:18native pairs, two real contexts,6CDF and2LUT controls, completion upper396.03s within20pair/45min cap. Benchmark stage:4dense+4rejection logical runs containing20native pairs,157.45s within20pair/30min cap. No additional retries. Post-exit audit verifies687 registered files and all8 complete benchmark runs.

The common target is m_i*E31(delta_i)/Z, p44 alias mass total2^62. Sampler TV is<=2.042e-13 rejection failure /8.237e-15 denseCDF relative to this finite law. E31 adds separate model TV<=1.721e-8 vs idealqexp(delta); prior F16/PWL, q-compilation and backend failure/security terms are separate. TCC2022 already covers the general rejection skeleton; SPRINT/CRYPTPEFT are strong model co-design priors. Actual whole-implementation security/concreteFABLE failure reduction and independent novelty remain unfinished.

## Reproduction boundary

Own source, registrations, backend patch/recipe and prior-source manifest are copied verbatim. Public result JSONs are explicitly reduced summaries without individual private audit hashes/logs. Full originals, raw shares/coins/token outputs, model captures and native binaries are frozen in the NEWSMY checkpoint. Large model/data/env assets and third-party PDFs/HTML are local only. A fresh clone lacks these fixtures and pinned builds; scripts/recipe expect the prior NEWSMY layout, not a clean installer. Apply existing backend/smallbatch patches before the new isolated18-bit variant. No upstream experimental reproduction is claimed.
