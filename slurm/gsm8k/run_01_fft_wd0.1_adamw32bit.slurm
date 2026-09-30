#!/bin/bash
#SBATCH --job-name=gsm8k-prompting-fft-adamw32bit

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generous vs the usual 8h: paged_adamw_32bit's fp32 optimizer states
# project to ~79GB peak VRAM on an 80GB A100 (measured 8-bit baseline
# 36.8GB + ~42GB for full-precision states) -- expect some CPU paging
# and a real (but likely well under 10x) speed penalty. 16h covers a
# pessimistic 10x slowdown on this dataset's ~1666s baseline training
# time with room to spare.
#SBATCH --time=16:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/gsm8k/prompting_fft_wd0.1_adamw32bit/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/prompting_fft_wd0.1_adamw32bit/
mkdir -p results/gsm8k/prompting_fft_wd0.1_adamw32bit/logs/

python -m src.experiments.runner \
configs/gsm8k/01_fft_wd0.1_adamw32bit.yaml \
$SLURM_ARRAY_TASK_ID
