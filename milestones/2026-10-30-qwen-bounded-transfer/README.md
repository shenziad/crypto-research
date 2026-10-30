# Fixed Qwen0.5B bounded-head transfer

The fixed larger-model transfer improved validation NLL by0.1253nat but retained48.7765% of the matched linear-control gain, missing the preregistered50% gate. The test corpora were not evaluated. This exact configuration is archived without parameter retries or new crypto implementation.

See [outcome and limits](notes/20261030_bounded_adapter_qwen_transfer_outcome.md), [frozen preregistration](notes/20261030_bounded_adapter_qwen_transfer_preregistration.json), and [complete training/validation records](results/bounded_adapter_qwen_transfer.json). Source is included. Model, data, environment, own trained weights and logs remain on NEWSMY; this is not a fresh-clone installer.

The earlier Pythia matched finite-E31 function/cost results remain valid. Qwen hidden896/vocab151936 and observed normalized-input L2 above20 are outside that crypto implementation and its range certificate. No test quality, full-model private inference, independent novelty or CCF-level claim.
