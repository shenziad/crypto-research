#!/usr/bin/env bash
set -euo pipefail
MILESTONE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$MILESTONE_ROOT"
mkdir -p toolchain results
for MILESTONE_NATIVE in dpf_aes_native dcf_wide_native dcf_sparse_native dcf_spline_native dcf_selector_native dcf_dither_native; do
  c++ -O3 -std=c++17 -maes -msse2 -fPIC -shared "experiments/$MILESTONE_NATIVE.cpp" -o "toolchain/$MILESTONE_NATIVE.tmp.so"
  mv "toolchain/$MILESTONE_NATIVE.tmp.so" "toolchain/$MILESTONE_NATIVE.so"
done
