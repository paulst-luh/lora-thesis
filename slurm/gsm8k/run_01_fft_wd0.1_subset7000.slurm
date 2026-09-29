#!/bin/bash
#SBATCH --job-name=gsm8k-prompting-fft-subset7000

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Only ~1.4x the data at the same 3 epochs (GSM8K's entire training
# pool is 7473 examples) -- weak diagnostic, included for consistency.
# Measured baseline here is ~1666s, so ~2330s worst case -- 8h leaves
# ample margin.
#SBATCH --time=08:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/gsm8k/prompting_fft_wd0.1_subset7000/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/prompting_fft_wd0.1_subset7000/
mkdir -p results/gsm8k/prompting_fft_wd0.1_subset7000/logs/

python -m src.experiments.runner \
configs/gsm8k/01_fft_wd0.1_subset7000.yaml \
$SLURM_ARRAY_TASK_ID
