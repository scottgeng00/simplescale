#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --job-name=arv2-vref
#SBATCH --output=/home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/arv2/logs/embed-%j.out
cd /home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/arv2
/home/sgeng/scottg/pretrain/lingua-clean/.venv-kd/bin/python arv2_vendi_ref.py
