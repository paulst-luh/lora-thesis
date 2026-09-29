#!/bin/bash
#SBATCH --job-name=conll2003-prompting-fft-adamw32bit

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generous vs the usual 8h: paged_adamw_32bit's fp32 optimizer states
# project to ~74GB peak VRAM on an 80GB A100 (measured 8-bit baseline
# 32.2GB + ~42GB for full-precision states) -- expect some CPU paging
# and a real (but likely well under 10x) speed penalty. 16h covers a
# pessimistic 10x slowdown on this dataset's ~909s baseline training
# time with room to spare.
#SBATCH --time=16:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/conll2003/prompting_fft_wd0.1_adamw32bit/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/conll2003/prompting_fft_wd0.1_adamw32bit/
mkdir -p results/conll2003/prompting_fft_wd0.1_adamw32bit/logs/

python -m src.experiments.runner \
configs/conll2003/01_fft_wd0.1_adamw32bit.yaml \
$SLURM_ARRAY_TASK_ID
