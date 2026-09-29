#!/bin/bash
#SBATCH --job-name=conll2003-prompting-main-attention-riskyseeds

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-6%4

#SBATCH --output=results/conll2003/prompting_main_attention_riskyseeds/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/conll2003/prompting_main_attention_riskyseeds/
mkdir -p results/conll2003/prompting_main_attention_riskyseeds/logs/

python -m src.experiments.runner \
configs/conll2003/11_main_attention_riskyseeds.yaml \
$SLURM_ARRAY_TASK_ID
