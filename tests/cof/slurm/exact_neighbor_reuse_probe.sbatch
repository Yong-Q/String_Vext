#!/bin/bash -l
#SBATCH --job-name=vext_exact_reuse_probe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.exact_neighbor_reuse_probe \
  --prepared /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_string_oldff_v1/prepared_smoke100 \
  --pilot-report /home/qiuyong/MOF-uni-MOF/diffusion/inputs/cof8948_oldff_vext_pilot_v1/report.json \
  --output /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_vext_exact_reuse_probe_v1 \
  --centers 128
