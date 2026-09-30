#!/bin/bash
#SBATCH --job-name=gsm8k-fft

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-17%4

#SBATCH --output=results/gsm8k/fft/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/fft/
mkdir -p results/gsm8k/fft/logs/

python -m src.experiments.runner \
configs/gsm8k/fft.yaml \
$SLURM_ARRAY_TASK_ID