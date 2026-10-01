#!/bin/bash -l
#SBATCH --job-name=vext_exact_native_fullgrid
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
module load gcc/13.1.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_vext_native_probe_v4
g++ -O3 -std=c++17 -fPIC -shared -nostdlib \
  /home/qiuyong/MOF-uni-MOF/diffusion/native/vext_exact_reuse_v1/kernel.cpp \
  -o /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_vext_native_probe_v4/kernel.so
/home/qiuyong/.conda/envs/moformer/bin/python -B -m graph_vext.exact_neighbor_reuse_fullgrid \
  --name bne+C56+C32+L14_COOH --shard shard0 \
  --native-library /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_vext_native_probe_v4/kernel.so \
  --output /home/qiuyong/MOF-uni-MOF/diffusion/runs/cof8948_oldff_vext_native_probe_v4/result
