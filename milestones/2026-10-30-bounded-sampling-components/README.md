# Bounded private output sampling: completed component milestone

Partial constructive research result, 2026-10-30. No complete sampler, end-to-end speedup, new primitive or CCF C+ claim. The model-invariance route remains a documented negative result.

Read [the mathematical contract, quality checks, prior work and limits](notes/20261030_bounded_sampling_finite_components.md).

- Frozen Pythia14M bounded rank8 output adaptation has positive quality evidence. One fixed F16/F20 realization passes a conditional rational range/error certificate and reused3072position conformance; these reused windows are not fresh blind quality evidence.
- Classical finite descending-uniform Bernoulli with p44/L19/K201 has a separately bounded sampling error2.6430181366e-13 relative to the exact finiteq/delta contract. It is not a bound on total model approximation or implementation security.
- Actual secret acceptance/select:7pairs passed; four201candidate cases have identical traffic23047717B and375669 exactANDs.
- Actual private Bx/Arow dot/floor/tanh plus acceptance/select within one existing SCI GC:6pairs passed. Independent Pythonbigint audit checks1050intermediate values and2token-only results; full201token-only traffic194986693B. Selected rows are still harness-supplied, not a connected oblivious lookup.
- Pinned FABLE512backend:4alternating paired runs pervariant,2048exactwide rows; medians312806052B/1615.5ms versus default8192buckets3885593637.5B/7510.455ms. Local fixed-build comparison only; original compile/boundary failures remain archived locally.

The next required step is actual secret alias -> server integerA rowlookup -> GC, with each party passing only its own shares and no parent reconstruction between phases. Then measure complete costs and compare strong dense/sampling baselines. Original FABLE row fixtures use FP32 bytes and cannot be silently reinterpreted as fixed-point coefficients.

## Reproduction scope

The pure mathematical check needs only Python standard library. To replay without overwriting archived results, create a scratch directory on NEWSMY with empty `results/`, copy `experiments/finite_descending_bernoulli.py` and `experiments/finite_descending_bernoulli_check.py` to its `experiments/`, and execute the latter. Compare the resulting JSON to the archived result.

The remaining scripts are the exact local research sources, not a clean-clone installer. GC builds require the pinned FABLE/SCI sources, layered patches and build environment recorded under `sources/`, plus the original local directory layout or deliberate path adaptation. Model checks require the official model revision/hash, public corpora hashes and locally archived own adapter NPZ/captures specified in preregistrations. They are not bundled here. Binary environments, transcript sharefiles, large model weights, corpus text and third-party PDFs remain on NEWSMY. Source/binary/input hashes make these local records reviewable; hashes alone are not evidence that a fresh machine reproduces every benchmark.

Initial misleading SCI gate-counter records are preserved locally in `sources/checkpoints/20261030_gc_initial_counter_audit`; v2 changes metrics only and reruns all13pairs. SCI `num_and()` reports a rounded hash-key counter; exactANDs here derive from compute-stage halfgate table bytes/32.
