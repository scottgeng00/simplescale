#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
cd /home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/arv2
/home/sgeng/scottg/pretrain/lingua-clean/.venv-kd/bin/python embed_eval.py
