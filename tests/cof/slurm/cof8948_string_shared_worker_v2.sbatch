#!/bin/bash -l
#SBATCH --job-name=cof8948_shared
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --exclude=gpu3
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
module load gcc/13.1.0 cuda/11.8
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SLOT:?set the six-worker slot index 0..5}"
/home/qiuyong/.conda/envs/moformer/bin/python -B -c '
import json
from pathlib import Path
path = Path("/home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_string_oldff_v1/shared_pool_v2/initialization_report.json")
gate = json.loads(path.read_text())
if not gate.get("passed") or gate.get("task_count") != 53688:
    raise SystemExit("shared task pool initialization gate failed")
'
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_string_shared_v2 \
  --mode run --slot "${SLOT}" --max-concurrent 25 --stagger 5
