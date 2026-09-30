#!/bin/bash
#SBATCH --job-name=mnli-prompting-main-attention

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-71%4

#SBATCH --output=results/mnli/prompting_main_attention/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_main_attention/
mkdir -p results/mnli/prompting_main_attention/logs/

python -m src.experiments.runner \
configs/mnli/11_main_attention.yaml \
$SLURM_ARRAY_TASK_ID
