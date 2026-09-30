#!/bin/bash
#SBATCH --job-name=gsm8k-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-53%4

#SBATCH --output=results/gsm8k/full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/full/
mkdir -p results/gsm8k/full/logs/

python -m src.experiments.runner \
configs/gsm8k/full.yaml \
$SLURM_ARRAY_TASK_ID