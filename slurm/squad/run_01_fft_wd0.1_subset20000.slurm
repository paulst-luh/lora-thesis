#!/bin/bash
#SBATCH --job-name=squad-prompting-fft-subset20000

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 4x the data at the same 3 epochs -> roughly 4x the training time.
# Measured baseline here is ~1679s, so ~6716s worst case -- 8h leaves
# ample margin.
#SBATCH --time=08:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/squad/prompting_fft_wd0.1_subset20000/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_fft_wd0.1_subset20000/
mkdir -p results/squad/prompting_fft_wd0.1_subset20000/logs/

python -m src.experiments.runner \
configs/squad/01_fft_wd0.1_subset20000.yaml \
$SLURM_ARRAY_TASK_ID
