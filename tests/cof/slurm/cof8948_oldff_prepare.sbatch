#!/bin/bash -l
#SBATCH --job-name=cof8948_oldff_prepare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${OUTPUT:?set distinct output directory}"
ARGS=(--output "$OUTPUT")
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_oldff_prepare "${ARGS[@]}"
