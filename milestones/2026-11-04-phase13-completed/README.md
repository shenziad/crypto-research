# Phase 13 completed paired confirmation

All 160 generation rows and 240 teacher-evaluation rows passed independent checks of input partitions, source hashes, training-weight provenance and metrics. There are 16 distinct generation inputs, with seeds retained within each input's paired bootstrap cluster, and 48 fresh teacher inputs. Generation took 904.19 seconds. No new native private calls were made.

| Prompt / method | Literal name/place recall | EOS rate | ROUGE-L |
|---|---:|---:|---:|
| Flat public | 70.97% | 18.75% | 0.2464 |
| Flat bounded, two-seed mean | 66.13% | 40.63% | 0.2591 |
| Flat linear, two-seed mean | 85.48% | 65.63% | 0.3315 |
| Two-shot public | 61.29% | 50.00% | 0.3051 |
| Two-shot bounded, two-seed mean | 69.35% | 90.63% | 0.3228 |
| Two-shot linear, two-seed mean | 75.81% | 90.63% | 0.3250 |

The public-prompt diagnostic coverage improvement did not replicate: public two-shot minus flat recall is -9.68 percentage points (exploratory paired interval -28.13 to +9.38). Against its own public two-shot baseline, bounded adaptation adds +8.06 points in recall (interval -9.38 to +26.56), +40.63 in EOS (interval +15.63 to +65.63), and +0.0177 ROUGE-L (interval -0.0163 to +0.0490). Both bounded heads worsen teacher-forced named-token NLL despite lower overall NLL. Termination improved; factuality repair is not established.

Bounded minus linear two-shot recall is -6.45 points (interval -20.97 to +6.67). An interval crossing zero does not establish equivalence or no quality loss. The two bounded seeds achieve 24/31 and 19/31 entity mentions. One public two-shot output and one output from the second bounded seed contain a demonstration entity absent from the target attributes. All rejection-failure counts are zero; no non-EOS special token IDs were observed. Both heuristic slot-error measures and all negative results are retained in the aggregate file; neither heuristic is a ground-truth factuality score.

Intervals use 20,000 paired input-cluster percentile resamples and are exploratory, without multiplicity correction. This is a central floating-CDF reference of specified finite weights/failure law, not the interactive protocol or exact integer-CDF execution. The lost original watchdog was reattached to the same process; completion was saved and exit observed, but the monitoring gap prevents certification of a whole-run resource peak or process exit code.

No additional expensive private trace is launched on the basis of this factuality result. Security work retains explicit conditional lemmas and open actual-sampler RLWE, full ciphertext/GC/OT/software and independent-review obligations. MoE additional full-size hybrid runs remain paused; C10 remains stopped. Source, raw generations, weights and party records remain local pending disclosure authorization.

A fixed posthoc teacher replay reproduced all 240 saved losses exactly. All 48 EOS positions have raw magnitude above the bound in every bounded head, while both bounded two-shot heads slightly worsen named-token NLL. This supports examining the objective and termination tradeoff; it does not prove a causal explanation of free-running omissions. No training or parameter search was performed in this diagnostic.
