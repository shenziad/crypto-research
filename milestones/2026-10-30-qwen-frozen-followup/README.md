# Frozen Qwen follow-up: positive quality signals continue

The prior fixed50% validation gate remains missed. On user instruction to improve both positive signals, a separately prospectively registered follow-up evaluates the same frozen weights on the original test windows, without retraining or any hyperparameter/checkpoint changes.

Across32 reset windows/4096 predictions each, bounded-minus-base NLL is-0.12210977 on PTB and-0.00626282 on Wiki; descriptive window intervals are[-0.13632,-0.10761] and[-0.01141,-0.00128]. The Wiki gain is small. Retained linear-control gains are44.77%/26.33%, not50%. These observations justify continuing evaluation, not declaring generation or CCF-level success.

See the [decision correction](notes/20261030_positive_signal_decision_correction.md), [outcome](notes/20261030_qwen_frozen_holdout_outcome.md), [complete records](results/bounded_adapter_qwen_frozen_holdout.json), and [conditional range certificate](results/qwen_bounded_head_range_certificate.json).

The new range contract uses declared public inputL2<=52, B-rowL2<=9/4 and A-rowL1<=3/2 caps, and retains F16/F20. It proves conditional finite ranges and a uniform error bound, not CUDA all-input correctness or actual security. Qwen still requires real896/151936 native implementation,18-bit secret indices/output and95-bit weightedCDF withp44 alias. No Qwen crypto speed result is claimed.

Own source/preregistration/results are preserved verbatim. Model/data/weights/captures/logs and original checkpoint stay on NEWSMY. This is not a standalone fresh-clone installer: scripts expect pinned dependencies and private local fixtures from earlier milestones. Fullfinite quality conformance, actual same-function private benchmarks, novelty and joint implementation security remain work in progress.
