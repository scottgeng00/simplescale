#!/bin/bash
#SBATCH --job-name=mix16-vendi
#SBATCH --output=/home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/promptmix/logs/vendi-%j.out
#SBATCH --account=dream
#SBATCH --partition=h100_mdf
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
cd /home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/promptmix
/home/sgeng/scottg/pretrain/lingua-clean/.venv-kd/bin/python pilot_vendi.py --out pilot_vendi.json
