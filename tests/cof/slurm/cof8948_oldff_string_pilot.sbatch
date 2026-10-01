#!/bin/bash -l
#SBATCH --job-name=cof8948_oldff_string_pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu3
#SBATCH --chdir=/home/qiuyong/MOF-uni-MOF/diffusion
#SBATCH --output=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.out
#SBATCH --error=/home/qiuyong/MOF-uni-MOF/diffusion/logs/%x_%j.err
set -euo pipefail
module load gcc/13.1.0 cuda/11.8
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.cof8948_string_runner \
  --mode pilot \
  --prepared /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_string_oldff_v1/prepared_v1 \
  --output /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_string_oldff_v1/pilot_v1
