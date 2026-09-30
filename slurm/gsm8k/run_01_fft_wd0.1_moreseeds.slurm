#!/bin/bash
#SBATCH --job-name=gsm8k-prompting-fft-moreseeds

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-20%4

#SBATCH --output=results/gsm8k/prompting_fft_wd0.1_moreseeds/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/prompting_fft_wd0.1_moreseeds/
mkdir -p results/gsm8k/prompting_fft_wd0.1_moreseeds/logs/

python -m src.experiments.runner \
configs/gsm8k/01_fft_wd0.1_moreseeds.yaml \
$SLURM_ARRAY_TASK_ID
