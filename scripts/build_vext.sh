#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cxx="${CXX:-g++}"
output="${VEXT_OUTPUT:-${root}/build/vext_kernel.so}"
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite ${output}" >&2
  exit 1
fi
mkdir -p "$(dirname "${output}")"
"${cxx}" -O3 -std=c++17 -fPIC -shared -nostdlib \
  "${root}/native/vext/kernel.cpp" -o "${output}"
sha256sum "${root}/native/vext/kernel.cpp" "${output}"
