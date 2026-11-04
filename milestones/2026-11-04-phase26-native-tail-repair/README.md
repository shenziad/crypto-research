# Phase 26 — integrated native tail-read repair

The active research backend is now **phase26-cot-20261104**, after the source, diagnostic, native-fault and complete-functional gates passed. This is an implementation correction to the existing protocol; parameters, finite arithmetic, distributions and output/leakage contract are unchanged. The fresh-label/direct-NP and CBD alternatives were not adopted. Old sources, profile, binaries and all negative evidence remain preserved.

A complete IKNP sender census expanded the repair to five postprocessors: block GOT, packed GOT, COT, ROT and unsigned modulo-add COT. Only their valid-prefix hash counts changed in one header. Both FABLE variants and all five actual native backends were rebuilt, with compile dependencies, entropy flags, runtime protection and19 BN return checks audited. The first COT-only build is explicitly superseded, not silently relabeled.

Primary COT covers1024 tail remainders after0/1/16 full blocks (3072 lengths,12288 method invocations per diagnostic build). The five sender/receiver methods cover10240 all-tail method/length cases plus176 width/active-length cases. Valid outputs, receiver relations, initialized-prefix read counts and full wire bytes agree with a defined public-poison reference. The repaired extracted methods do not add blanket pad initialization; they avoid requesting reads of the unused suffix. This supports a local source lemma against a defined reference, not C++ equivalence to undefined original execution.

Optimized and ASan+UBSan diagnostics pass. An initial LeakSanitizer run could not ptrace a nondumpable fixture and its failure is retained. A separate entirely public fixture, omitting the runtime guard only in that diagnostic program, passed ASan+UBSan+LeakSanitizer on the host. Real native/worker protections stay enabled. This is not MSan coverage or whole-native/dependency leak-freedom.

All25 fixed native fault cases pass the expected failure/no-final-output checks. Two preconnect dense cases use allowed cancellation of an owned peer. One complete repaired candidate run and one same-function dense run pass independent integer/row/arithmetic/token audits with fresh randomness:1539062885 bytes /38.477s and3508012338 bytes /63.296s. These are regression observations, not new performance conclusions; the small candidate byte variation is not attributed as a repair benefit. No affected old whole protocol was executed.

The validation peak owned RSS was4472729600 bytes with no budget event, on two low-priority CPUs and no GPU. No KV task was modified. No software installed, sealed tests opened, task/seed/parameter sweep or model training. MoE/C10 remain paused.

The specific tail-read issue is repaired in the selected research implementation. Full GC/OT/common-AES simulation, actual-law RLWE and known-key whole-cipher consistency remain unclosed; no exploitable leakage or overall security bits are asserted. Phase25's direct-NP cost penalties and unestablished private-adapter task value remain negative evidence. Continue proof/model review and cost-gated research without relabeling these tests as a cryptographic theorem.

Only aggregate research information is published. Detailed source, patches, party logs, weights and keys remain local pending explicit export authorization.
