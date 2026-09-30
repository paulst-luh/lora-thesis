#!/bin/bash
#SBATCH --job-name=squad-prompting-fft-adamw32bit

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generous vs the usual 8h: paged_adamw_32bit's fp32 optimizer states
# project to ~82GB peak VRAM here -- squad already has the highest
# baseline VRAM of any dataset (40.1GB measured with the 8-bit
# optimizer), so this is the one most likely to exceed 80GB and need
# real CPU paging. 16h covers a pessimistic 10x slowdown on this
# dataset's ~1679s baseline training time with room to spare -- but
# watch this one specifically; if it's still running near the limit,
# that's a sign paging is worse than expected, not just a fluke.
#SBATCH --time=16:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/squad/prompting_fft_wd0.1_adamw32bit/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_fft_wd0.1_adamw32bit/
mkdir -p results/squad/prompting_fft_wd0.1_adamw32bit/logs/

python -m src.experiments.runner \
configs/squad/01_fft_wd0.1_adamw32bit.yaml \
$SLURM_ARRAY_TASK_ID
