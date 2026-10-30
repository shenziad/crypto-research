# Matched finite-E31 private output-head sampling

Complete pinned research implementations for the same finite probability law, with client-only final token output. Four alternating paired comparisons all passed: dense 1,030,608,508 B / 7,307.45 ms median whole-pair wall; selected rejection 633,235,406.5 B / 4,304.29 ms. Observed reductions 38.56% bytes / 41.10% wall.

See [outcome and limitations](notes/20261030_matched_exp31_outcome.md), [finite functionality and conditional composition](notes/20261030_matched_exp31_functionality.md), and [retrospective audit summary](results/matched_exp31_closure_audit.json).

The E31 profile is supplementary to the prior p44 descending-chain profile. Relative to this exact finite target both sampling errors are below 2^-40; E31 adds a separate model TV bound of 1.721e-8 versus ideal q exp(delta). The original certificate's 92-bit CDF metadata and copied source opening comment are historical; the implemented dense sampler uses 93 bits and includes final sampling. The outcome explains these errata without modifying frozen files.

This is not a new primitive, full private LLM, global SOTA comparison, actual implementation security proof, or CCF-level novelty claim. TCC2022 already supplies the rejection framework. Full same-function comparison is now complete; model/task generalization, concrete FABLE failure bounds, and joint security remain open.

## Reproduction boundary

Own source and preregistrations are preserved verbatim. Public result JSONs are explicitly derived summaries: private audit-file hashes and logs are omitted. Full frozen originals, own model fixtures, all audit coins/shares, outputs and native binaries are in the NEWSMY checkpoint; they are not published here. A fresh clone cannot reproduce real-context benchmarks without these fixtures and pinned dependencies. Do not interpret preregistration hashes of local files as supplied public assets.

Build and execution require the previously pinned FABLE/SCI, libOTe, SEAL, compiler and Python dependencies described in the [previous full-chain milestone](../2026-10-30-bounded-sampling-fullchain/README.md). New build drivers are under `experiments/`; they target the NEWSMY layout. These are research scripts, not a new fresh-clone installer. The retrospective audit requires original local raw files, so cannot run against the reduced public summaries. No unpublished third-party paper text, corpus or large weights are bundled.
