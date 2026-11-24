#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3:00:00
set -euo pipefail
cd /home/sgeng/scottg/pretrain/lingua-clean
.venv-kd/bin/python analysis/multisample_vendi.py --out analysis/multisample_vendi.json
