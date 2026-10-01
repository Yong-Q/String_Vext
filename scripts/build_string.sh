#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcc="${NVCC:-nvcc}"
cxx="${CXX:-g++}"
arch="${CUDA_ARCH:-sm_89}"
output="${STRING_OUTPUT:-${root}/build/string_triclinic_cuda}"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite ${output}" >&2
  exit 1
fi
mkdir -p "$(dirname "${output}")"
"${nvcc}" -std=c++17 -O2 -ccbin "${cxx}" -arch="${arch}" \
  "${root}/native/string/source_code/RC_main.cu" -lcufft \
  -o "${output}"
sha256sum "${root}/native/string/source_code/RC_main.cu" \
  "${output}"
