#!/bin/bash
#SBATCH --job-name=mnli-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-53%4

#SBATCH --output=results/mnli/full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/full/
mkdir -p results/mnli/full/logs/

python -m src.experiments.runner \
configs/mnli/full.yaml \
$SLURM_ARRAY_TASK_ID