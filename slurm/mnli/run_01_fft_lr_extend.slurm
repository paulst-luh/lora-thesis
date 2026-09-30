#!/bin/bash
#SBATCH --job-name=mnli-prompting-fft-lr-extend

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-11%4

#SBATCH --output=results/mnli/prompting_fft_lr_extend/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_fft_lr_extend/
mkdir -p results/mnli/prompting_fft_lr_extend/logs/

python -m src.experiments.runner \
configs/mnli/01_fft_lr_extend.yaml \
$SLURM_ARRAY_TASK_ID
