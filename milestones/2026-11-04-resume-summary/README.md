# Research restart checkpoint — 2026-11-04

The resumed bounded-head training completed all64 fixed updates and retained recoverable checkpoints. On48 newly fixed test MRs and16 generation MRs, named/EOS weighting did not establish a factuality repair: both bounded weighted heads still had worse named-token NLL than the public base, and lower named recall than their bounded CE comparators. Linear comparators remained stronger. All failures and both diagnostic rule sets are retained; NLL is not factuality.

A new64-call private execution on the prospectively selected head/MR is running. Ten public completion steps had been observed at this checkpoint; correctness audits wait until both complete party sessions exit. Four fresh dense snapshots are planned. This is not yet a completed64-call result or security theorem.

MoE streaming passed10 synthetic and15 real matched case pairs per version against independent integer references and the original implementation. Version1 added cost; version2 restores cache reuse only for unchanged operands across tiles and applies it equally to every strong baseline. Actual D128/H128 and D256/H128 crops retain communication savings on the fixed non-repeated batch. Repeated-input q=64 fallback matches the same-q baseline but remains costlier than the fully hidden strong baseline. Full shape has not been run.

Security review remains conditional, including known-key response privacy, complete proposal views, GC/OT composition and fixed stopping behavior. No full-protocol128-bit proof, full-model quality, WAN or external SOTA claim is made. C10 remains paused.

This checkpoint publishes aggregate findings only. Detailed source and party records remain local pending explicit public-disclosure authorization.
