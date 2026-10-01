#!/bin/bash -l
#SBATCH --job-name=cof8948_oldff_vext_pilot
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_oldff_vext \
  --mode pilot \
  --prepared /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_string_oldff_v1/prepared_smoke100 \
  --output /home/qiuyong/MOF-uni-MOF/diffusion/inputs/cof8948_oldff_vext_pilot_v1
