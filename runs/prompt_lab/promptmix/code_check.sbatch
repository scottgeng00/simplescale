#!/bin/bash
#SBATCH --job-name=mix16-codecheck
#SBATCH --output=/home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/promptmix/logs/codecheck-%j.out
#SBATCH --account=dream
#SBATCH --partition=h100_mdf
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# CPU-only execution check of prompt_pool code views: sbatch --export=ALL,RUN=...,OUT=...[,EXTRA=...] code_check.sbatch
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd /home/sgeng/scottg/pretrain/simplescale
which gcc; gcc --version | head -1; which javac || echo "no javac"; /usr/bin/python3 --version
.venv/bin/python scripts/check_mix16_code.py --run "$RUN" --out "$OUT" --workers 60 ${EXTRA:-}
