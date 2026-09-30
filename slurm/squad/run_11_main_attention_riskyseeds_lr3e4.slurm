#!/bin/bash
#SBATCH --job-name=squad-prompting-main-attention-riskyseeds-lr3e4

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-6%4

#SBATCH --output=results/squad/prompting_main_attention_riskyseeds_lr3e4/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_main_attention_riskyseeds_lr3e4/
mkdir -p results/squad/prompting_main_attention_riskyseeds_lr3e4/logs/

python -m src.experiments.runner \
configs/squad/11_main_attention_riskyseeds_lr3e4.yaml \
$SLURM_ARRAY_TASK_ID
