#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=6:00:00
set -euo pipefail
cd /home/sgeng/scottg/pretrain/lingua-clean
.venv-kd/bin/python setup/export_multisample.py --run $RUN --out "$OUT" --pool-mult 0.25 --order sample-major --workers 60
