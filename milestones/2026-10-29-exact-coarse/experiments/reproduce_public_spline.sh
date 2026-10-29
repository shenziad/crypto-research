#!/usr/bin/env bash
# All generated artifacts and caches stay in this NEWSMY project.
set -euo pipefail
SPLINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SPLINE_CPU_PYTHON="$SPLINE_ROOT/envs/torch-cpu/bin/python"
SPLINE_GPU_PYTHON=${SPLINE_GPU_PYTHON:-/media/shenziad/NEWSMY/KVcacheSC/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$SPLINE_ROOT/toolchain/hf-cache" XDG_CACHE_HOME="$SPLINE_ROOT/toolchain/xdg-cache"
export CUDA_CACHE_PATH="$SPLINE_ROOT/toolchain/cuda-cache" TORCH_HOME="$SPLINE_ROOT/toolchain/torch-cache"
export TRITON_CACHE_DIR="$SPLINE_ROOT/toolchain/triton-cache" PYTORCH_KERNEL_CACHE_PATH="$SPLINE_ROOT/toolchain/torch-kernel-cache"
export MPLCONFIGDIR="$SPLINE_ROOT/toolchain/matplotlib-cache"
mkdir -p "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH" "$MPLCONFIGDIR"
cd "$SPLINE_ROOT"
SPLINE_MODE=${1:---checks-only}
if [[ "$SPLINE_MODE" != --checks-only && "$SPLINE_MODE" != --full ]]; then
  echo 'Usage: bash experiments/reproduce_public_spline.sh [--checks-only|--full]' >&2
  exit 2
fi
for SPLINE_BACKEND in dcf_wide_native dcf_spline_native dcf_selector_native; do
  c++ -O3 -std=c++17 -maes -msse2 -fPIC -shared "experiments/$SPLINE_BACKEND.cpp" -o "toolchain/$SPLINE_BACKEND.tmp.so"
  mv "toolchain/$SPLINE_BACKEND.tmp.so" "toolchain/$SPLINE_BACKEND.so"
done
for SPLINE_BACKEND in dcf_cuda dcf_cuda_protocol; do
  toolchain/cuda128/bin/nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler -fPIC --shared "experiments/$SPLINE_BACKEND.cu" -o "toolchain/$SPLINE_BACKEND.tmp.so"
  mv "toolchain/$SPLINE_BACKEND.tmp.so" "toolchain/$SPLINE_BACKEND.so"
done
"$SPLINE_CPU_PYTHON" experiments/dcf_cuda_aes_reference.py
"$SPLINE_CPU_PYTHON" experiments/dcf_wide_native_check.py
"$SPLINE_CPU_PYTHON" experiments/public_spline_selector_check.py
"$SPLINE_CPU_PYTHON" experiments/public_polynomial_basis_check.py
"$SPLINE_GPU_PYTHON" experiments/dcf_cuda_check.py
"$SPLINE_GPU_PYTHON" experiments/dcf_cuda_protocol_check.py
if [[ "$SPLINE_MODE" == --full ]]; then
  "$SPLINE_CPU_PYTHON" experiments/qwen_public_spline_protocol.py --frontier
  "$SPLINE_CPU_PYTHON" experiments/qwen_public_spline_protocol.py --frontier --spawn
  "$SPLINE_CPU_PYTHON" experiments/qwen_public_spline_bandwidth.py
  "$SPLINE_GPU_PYTHON" experiments/qwen_public_spline_protocol.py --frontier --cuda
  "$SPLINE_GPU_PYTHON" experiments/qwen_public_spline_bandwidth.py --cuda
  "$SPLINE_GPU_PYTHON" experiments/qwen_public_spline_quality.py --extended-test
  "$SPLINE_GPU_PYTHON" experiments/qwen_public_spline_quality.py --corpus-path data/ptb_test.txt --evaluation-label ptb32x256 --windows 32 --stride 1024 --length 257
  "$SPLINE_CPU_PYTHON" experiments/public_spline_frontier_report.py
  "$SPLINE_CPU_PYTHON" experiments/public_spline_gpu_frontier_report.py
fi
