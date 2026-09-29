#!/bin/bash
#SBATCH --job-name=mnli-prompting-main-attention-riskyseeds-lr5e4

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-6%4

#SBATCH --output=results/mnli/prompting_main_attention_riskyseeds_lr5e4/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_main_attention_riskyseeds_lr5e4/
mkdir -p results/mnli/prompting_main_attention_riskyseeds_lr5e4/logs/

python -m src.experiments.runner \
configs/mnli/11_main_attention_riskyseeds_lr5e4.yaml \
$SLURM_ARRAY_TASK_ID
