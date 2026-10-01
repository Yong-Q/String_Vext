#!/bin/bash -l
#SBATCH --job-name=cof8948_vext_exact
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${RANK:?RANK must be 0 or 1}"
/home/qiuyong/.conda/envs/moformer/bin/python -B -c '
import json
from pathlib import Path
base = Path("/home/qiuyong/MOF-uni-MOF/diffusion/runs")
gates = (
    ("cof8948_oldff_vext_exact_pilot_v2/report.json", "gas_cases", 12),
    ("cof8948_vext_endtoend_benchmark_v1/report.json", "materials", 2),
    ("cof8948_vext_exact_audit50_v1/report.json", "materials", 50),
    ("cof8948_exact_string_path_audit50_v1/report.json", "paths", 300),
)
for path, key, expected in gates:
    record = json.loads((base / path).read_text())
    if not record.get("passed") or record.get(key) != expected:
        raise SystemExit(f"exact Vext preflight failed: {path}")
'
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_oldff_vext_exact \
  --mode run --shard "${RANK}" --workers "${SLURM_CPUS_PER_TASK}"
