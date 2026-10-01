#!/bin/bash -l
#SBATCH --job-name=build_legacy8238_v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/legacy8238_build_v4_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/legacy8238_build_v4_%j.err
set -euo pipefail
cd /home/qiuyong/MOF-uni-MOF/diffusion
module purge
module load gcc/13.1.0 cuda/11.8
export PATH=/usr/local/cuda-13.3/bin:"$PATH"
export PYTHONDONTWRITEBYTECODE=1
nvcc_bin="$(command -v nvcc || true)"
if [[ -z "$nvcc_bin" || ! -x "$nvcc_bin" ]]; then
    nvcc_bin=/usr/local/cuda-13.3/bin/nvcc
fi
if [[ ! -x "$nvcc_bin" ]]; then
    echo "nvcc unavailable at module path or fallback: $nvcc_bin" >&2
    exit 2
fi
gxx_bin="$(command -v g++)"
root=native/string_legacy8238_v4
exe="$root/GPU_string_legacy8238_v4"
test ! -e "$exe"
test ! -e "$exe.build.json"
buildlog="$root/build_${SLURM_JOB_ID}"
mkdir "$buildlog"
printf 'nvcc=%s\ng++=%s\n' "$nvcc_bin" "$gxx_bin" > "$buildlog/toolchain_paths.txt"
"$nvcc_bin" --version > "$buildlog/nvcc_version.txt"
"$gxx_bin" --version > "$buildlog/gcc_version.txt"
env | sort > "$buildlog/environment.txt"
sha256sum "$root"/source_code/* > "$buildlog/source_before.sha256"
python native/string_legacy8238_v1/test_host.py > "$buildlog/host_tests.log" 2>&1
"$nvcc_bin" -std=c++17 -O2 -ccbin "$gxx_bin" -arch=sm_89 \
    "$root/source_code/RC_main.cu" -lcufft -o "$buildlog/solver" \
    > "$buildlog/compile.out" 2> "$buildlog/compile.err"
sha256sum "$root"/source_code/* > "$buildlog/source_after.sha256"
cmp "$buildlog/source_before.sha256" "$buildlog/source_after.sha256"
ldd "$buildlog/solver" > "$buildlog/ldd.txt"
set +e
"$buildlog/solver" > "$buildlog/noargs.out" 2> "$buildlog/noargs.err"
status=$?
set -e
printf '%s\n' "$status" > "$buildlog/noargs.exit"
test "$status" -eq 2
ln "$buildlog/solver" "$exe"
sha256sum "$exe" > "$buildlog/exe.sha256"
python - "$exe" "$buildlog" <<'PY'
import hashlib,json,pathlib,sys
exe,logs=map(pathlib.Path,sys.argv[1:])
record=dict(executable=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
            source_manifest=str(logs/'source_before.sha256'),build_logs=str(logs),
            runtime_modules=['gcc/13.1.0','cuda/11.8'],cuda_version=(logs/'nvcc_version.txt').read_text(),
            validation='host_tests_and_noargs_only_not_optimizer_certificate')
with pathlib.Path(str(exe)+'.build.json').open('x') as f:
    json.dump(record,f,indent=2)
PY
