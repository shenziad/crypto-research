# Reproducibility and scope

This is a source and result archive of the completed finite gate/down study. It is not a fresh-machine full-model installer.

## Self-contained CPU functional smoke

Install Python with NumPy and cryptography, and an x86 compiler supporting AES-NI. From this milestone directory run:

```bash
bash scripts/build_cpu.sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/smoke_exact_coarse.py
```

The smoke uses the same frozen protocol implementation with small synthetic, discontinuous splines. It needs no model, corpus or CUDA. The original `experiments/exact_coarse_spline_check.py` provides the 8,388,608-pair exhaustive study; published results are preserved under `results/`.

## Model/block and CUDA reproduction

The `experiments/reproduce_exact_coarse.sh` script assumes the original project layout and prepared environments/assets. It is not a bootstrap installer. Full block runs require the pinned public models, fixed calibration, selected quantized matrices and captures listed in `external_asset_manifest.json`. These remain on NEWSMY; weights, corpus contents and third-party PDFs are excluded from Git. Model and dataset source manifests are included. Downloaders and numerical/capture generators are provided, but reconstructing the entire earlier preparation chain has not been verified on a clean machine. No claim of clean-machine end-to-end reproducibility is made yet.

The original script expects its CPU environment at `envs/torch-cpu/bin/python`; set `EXACT_GPU` to a suitable PyTorch CUDA Python. GPU tests require an NVIDIA GPU and the separately documented CUDA toolchain. Environment caches and compiled libraries are generated locally.

## Interpretation

The 432 executions are local finite gate/down blocks, with trusted dealer preparation and one GPU shared by both parties. Upstream projections, normalization and attention are outside that protocol. Published quality proxies keep those other operations FP32. The one-shot fixed-input ideal-DCF proof draft does not establish production implementation security or adaptive preprocessing. Prior-art restrictions are retained in the notes.

The refreshed aggregate includes both 4-window validation and completed 32-window 1.5B quality measurements; raw timing results are unchanged.

## Regenerate aggregate and scientific figure without model assets

With NumPy and Matplotlib installed, run `python3 experiments/exact_coarse_comparison_report.py` from this milestone directory. This reads the archived raw JSON results and regenerates the Pareto PNG/PDF and aggregate; it does not rerun timings or inference.
