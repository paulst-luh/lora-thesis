#!/bin/bash
#SBATCH --job-name=gsm8k-prompting-fft-wd0.1-epochs6

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 2x the epochs of every other FFT block here -> roughly 2x the wall-clock
# per run; bumped from 8h to 16h accordingly.
#SBATCH --time=16:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/gsm8k/prompting_fft_wd0.1_epochs6/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/prompting_fft_wd0.1_epochs6/
mkdir -p results/gsm8k/prompting_fft_wd0.1_epochs6/logs/

python -m src.experiments.runner \
configs/gsm8k/01_fft_wd0.1_epochs6.yaml \
$SLURM_ARRAY_TASK_ID
