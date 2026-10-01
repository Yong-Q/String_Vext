#!/bin/bash -l
#SBATCH --job-name=cof8948_lookup50
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_lookup_audit \
  --source /home/qiuyong/MOF-uni-MOF/diffusion/inputs/cof8948_oldff_vext_full_v1 \
  --output /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_lookup_audit50_v1 \
  --materials 50 --sample-centers 256 --wait-seconds 21600
