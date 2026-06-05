#!/bin/bash
#SBATCH --job-name=infrascanroad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=35:00:00
#SBATCH --output=infrascanroad_%j.log
#SBATCH --error=infrascanroad_%j.log

source /cluster/home/lkuehner/miniforge3/etc/profile.d/conda.sh
conda activate infrascan

cd /cluster/home/lkuehner/MSc_Thesis
echo "[$(date --iso-8601=seconds)] Starting infraScanRoad job ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u -m infraScan.infraScanRoad.main_pipeline
