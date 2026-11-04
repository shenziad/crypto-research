# Phase15: lexical viability development checkpoint

A fixed client-local viability policy and matched public/bounded/linear baselines were registered before generation. All80central-reference outputs on8fresh development inputs passed independent replay. All40constrained outputs included the specified entity patterns andEOS;3765finite-proposal support checks passed. Both preregistered mechanism and private-value development gates passed. No model was trained, no new native private protocol ran, and the previously sealed test remained unevaluated.

| Policy / method | Entity recall | EOS rate | ROUGE-L |
|---|---:|---:|---:|
| Unconstrained public |75.00%|50.00%|0.2918|
| Unconstrained bounded,two-seed mean |83.33%|87.50%|0.4247|
| Unconstrained linear,two-seed mean |70.83%|68.75%|0.3214|
| Viable public |100%|100%|0.2647|
| Viable bounded,two-seed mean |100%|100%|0.3862|
| Viable linear,two-seed mean |100%|100%|0.3260|

Coverage is supplied by the public constraint policy, not private adaptation. Bounded minus same-policy public ROUGE-L is+0.1215(exploratory paired8-input interval+0.0317to+0.2254); bounded minus linear is+0.0602(interval-0.0176to+0.1474), which does not establish superiority or equivalence. Importantly,the mask reduces bounded ROUGE-L by0.0385relative to its own unconstrained version and adds8.875tokens on average(posthoc descriptive contrast). Lexical inclusion therefore has a measured quality/cost tradeoff. Deadline-driven transitions are explicitly recorded. All failures and prior negative stages remain preserved.

There are3conflicting-input cases among these8selected inputs; earlier audits found11/128training and3/32guard-development cases with conflicting repeated fields. No selected inputs were replaced after inspection. Two slot-error heuristics are retained, but cannot certify all-attribute semantic truth. This small development signal is not a representative benchmark, full-model accuracy result or test confirmation. Central floating-CDF sampling of finite weights is not native protocol execution.

The full-sequence function and proposed leakage contract are now explicit. Local viability sampling changes the target law; it is not the original LM conditioned on eventual success. The earlier single-step protected-token counterexample remains. Source review covers duplicate-candidate remapping and model-independent ideal dummy outputs in the existing completed native trace; it does not prove the whole HE/GC/OT view or timing privacy.

Lexical constraints are established prior art:[GBS](https://aclanthology.org/P17-1141/),[NeuroLogic](https://aclanthology.org/2021.naacl-main.339/),and[conditional sampling viaMCMC](https://arxiv.org/abs/2506.05754). Public/private inference architectures also have prior art; no novelty claim follows from this screen. A potentially close[EDENpreprint](https://engrxiv.org/preprint/view/5468) requires full-text comparison; the canonical PDF was inaccessible and its detailed construction remains unverified. It is distinct from[Avitan etal.](https://aclanthology.org/2025.findings-ijcnlp.111/).

The fixed run completed in826.64seconds with sampled RSS3.02GiB and no resource-limit violation. Next work is fixed further validation of private utility on coherent tasks and independent security/prior-art review. No automatic test/native run follows this gate. MoE additional full-size experiments remain paused;C10 remains stopped. Only safe aggregates are published, with source/rawoutputs/weights/partyrecords retained locally pending disclosure authorization.
