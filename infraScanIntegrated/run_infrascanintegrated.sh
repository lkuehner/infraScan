#!/bin/bash
#SBATCH --job-name=infrascanintegrated
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --constraint=EPYC_9654
#SBATCH --time=35:00:00
#SBATCH --output=infrascanintegrated_%j.log
#SBATCH --error=infrascanintegrated_%j.log


source /cluster/home/lkuehner/miniforge3/etc/profile.d/conda.sh
conda activate infrascan

cd /cluster/home/lkuehner/MSc_Thesis
echo "[$(date --iso-8601=seconds)] Starting infraScanIntegrated job ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export INFRASCAN_MPL_BACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
mkdir -p "${MPLCONFIGDIR}"

python -u - <<'PY'
import builtins

answers = iter([
    "1",  # run mode: 1=Integrated
    "y",  # include standalone comparison outputs
    "",   # valuation year: keep integrated settings default
])

original_input = builtins.input

def auto_input(prompt=""):
    try:
        answer = next(answers)
    except StopIteration:
        answer = ""
    print(prompt + answer)
    return answer

builtins.input = auto_input

try:
    from infraScan.infraScanIntegrated.main_integrated import infrascan_integrated
    infrascan_integrated()
finally:
    builtins.input = original_input
PY
