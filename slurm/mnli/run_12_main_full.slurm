#!/bin/bash
#SBATCH --job-name=mnli-prompting-main-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-71%4

#SBATCH --output=results/mnli/prompting_main_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_main_full/
mkdir -p results/mnli/prompting_main_full/logs/

python -m src.experiments.runner \
configs/mnli/12_main_full.yaml \
$SLURM_ARRAY_TASK_ID
