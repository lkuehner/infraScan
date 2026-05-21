#!/bin/bash
#SBATCH --job-name=infrascanrail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=infrascanrail_%j.log

source /cluster/home/lkuehner/miniforge3/etc/profile.d/conda.sh
conda activate infrascan

cd /cluster/home/lkuehner/MSc_Thesis

python - <<'PY'
import builtins

answers = iter([
    "2",  # visualization mode: 1=manual, 2=none, 3=all
    "3",  # grouping strategy: 1=manual, 2=conservative, 3=baseline, 4=optimal
    "",   # capacity threshold -> default
    "",   # max iterations -> default
    "y",  # intervention costs reviewed
])

_original_input = builtins.input

def auto_input(prompt=""):
    try:
        answer = next(answers)
    except StopIteration:
        answer = ""
    print(prompt + answer)
    return answer

builtins.input = auto_input

from infraScan.infraScanRail.main_cap import infrascanrail_cap
infrascanrail_cap()
PY
