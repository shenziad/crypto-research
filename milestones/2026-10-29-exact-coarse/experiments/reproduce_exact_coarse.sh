#!/usr/bin/env bash
# Frozen project inputs are prerequisites; all outputs/caches stay on NEWSMY.
set -euo pipefail
EXACT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXACT_CPU="$EXACT_ROOT/envs/torch-cpu/bin/python"
EXACT_GPU=${EXACT_GPU:-/media/shenziad/NEWSMY/KVcacheSC/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$EXACT_ROOT/toolchain/hf-cache" XDG_CACHE_HOME="$EXACT_ROOT/toolchain/xdg-cache"
export CUDA_CACHE_PATH="$EXACT_ROOT/toolchain/cuda-cache" TORCH_HOME="$EXACT_ROOT/toolchain/torch-cache"
export TRITON_CACHE_DIR="$EXACT_ROOT/toolchain/triton-cache" PYTORCH_KERNEL_CACHE_PATH="$EXACT_ROOT/toolchain/torch-kernel-cache"
export MPLCONFIGDIR="$EXACT_ROOT/toolchain/matplotlib-cache"
cd "$EXACT_ROOT"
EXACT_MODE=${1:---checks-only}
if [[ "$EXACT_MODE" != --checks-only && "$EXACT_MODE" != --full ]]; then
  echo 'Usage: bash experiments/reproduce_exact_coarse.sh [--checks-only|--full]' >&2
  exit 2
fi
c++ -O3 -std=c++17 -maes -msse2 -fPIC -shared experiments/dcf_dither_native.cpp -o toolchain/dcf_dither_native.tmp.so
mv toolchain/dcf_dither_native.tmp.so toolchain/dcf_dither_native.so
"$EXACT_CPU" experiments/exact_coarse_spline_check.py
"$EXACT_GPU" experiments/exact_coarse_cuda_check.py
if [[ "$EXACT_MODE" == --full ]]; then
  for EXACT_MODEL in qwen05 qwen15; do
    "$EXACT_CPU" experiments/qwen_exact_coarse_compare.py --model "$EXACT_MODEL"
    "$EXACT_GPU" experiments/qwen_exact_coarse_compare.py --model "$EXACT_MODEL" --cuda
  done
  for EXACT_CORPUS in wiki ptb; do
    "$EXACT_GPU" experiments/qwen15_dither_quality.py --mode aligned --windows 32 --corpus "$EXACT_CORPUS"
  done
  "$EXACT_CPU" experiments/exact_coarse_comparison_report.py
fi
