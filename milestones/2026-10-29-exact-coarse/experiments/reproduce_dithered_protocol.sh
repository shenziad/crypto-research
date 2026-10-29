#!/usr/bin/env bash
# Reproduce the new branch from the project's pinned assets and frozen inputs.
set -euo pipefail
DITHER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DITHER_CPU="$DITHER_ROOT/envs/torch-cpu/bin/python"
DITHER_GPU=${DITHER_GPU:-/media/shenziad/NEWSMY/KVcacheSC/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$DITHER_ROOT/toolchain/hf-cache" XDG_CACHE_HOME="$DITHER_ROOT/toolchain/xdg-cache"
export CUDA_CACHE_PATH="$DITHER_ROOT/toolchain/cuda-cache" TORCH_HOME="$DITHER_ROOT/toolchain/torch-cache"
export TRITON_CACHE_DIR="$DITHER_ROOT/toolchain/triton-cache" PYTORCH_KERNEL_CACHE_PATH="$DITHER_ROOT/toolchain/torch-kernel-cache"
export MPLCONFIGDIR="$DITHER_ROOT/toolchain/matplotlib-cache"
cd "$DITHER_ROOT"
DITHER_MODE=${1:---checks-only}
if [[ "$DITHER_MODE" != --checks-only && "$DITHER_MODE" != --full ]]; then
  echo 'Usage: bash experiments/reproduce_dithered_protocol.sh [--checks-only|--full]' >&2
  exit 2
fi
c++ -O3 -std=c++17 -maes -msse2 -fPIC -shared experiments/dcf_dither_native.cpp -o toolchain/dcf_dither_native.tmp.so
mv toolchain/dcf_dither_native.tmp.so toolchain/dcf_dither_native.so
"$DITHER_CPU" experiments/dithered_classifier_reference_check.py
"$DITHER_CPU" experiments/dithered_spline_native_check.py
"$DITHER_CPU" experiments/dithered_silu_certificate.py
if [[ "$DITHER_MODE" == --full ]]; then
  # Generate the frozen calibration-derived dither capture first. No fitting to tests.
  "$DITHER_GPU" experiments/qwen_public_spline_quality.py --classifier-shift 8
  "$DITHER_CPU" experiments/qwen_dithered_protocol.py
  "$DITHER_GPU" experiments/qwen_dithered_protocol.py --cuda
  for DITHER_CORPUS in wiki ptb; do
    DITHER_ARGS=(--extended-test)
    if [[ "$DITHER_CORPUS" == ptb ]]; then
      DITHER_ARGS=(--corpus-path data/ptb_test.txt --evaluation-label ptb32x256 --windows 32 --stride 1024 --length 257)
    fi
    "$DITHER_GPU" experiments/qwen_public_spline_quality.py --classifier-shift 8 --exact-aligned "${DITHER_ARGS[@]}"
    for DITHER_SEED in 20261029 20261030; do
      "$DITHER_GPU" experiments/qwen_public_spline_quality.py --classifier-shift 8 --dither-seed "$DITHER_SEED" "${DITHER_ARGS[@]}"
    done
    "$DITHER_GPU" experiments/qwen_public_spline_quality.py --classifier-shift 8 --coarse-body "${DITHER_ARGS[@]}"
  done
  "$DITHER_CPU" experiments/dithered_protocol_report.py
fi
