#!/bin/bash
#SBATCH --job-name=mnli-prompting-ladder-isostep-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# iso_step fixes max_steps=938 at every rung, matching the main sweep's
# per-task compute profile -- bumped from the attention ladder's 8h to
# account for lora.target=full's ~35% slower per-run cost (1183s vs
# 876s mean, main sweep). 8h x 1.35 ~= 11h, rounded up to 12h.
#SBATCH --time=12:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/mnli/prompting_ladder_isostep_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_ladder_isostep_full/
mkdir -p results/mnli/prompting_ladder_isostep_full/logs/

python -m src.experiments.runner \
configs/mnli/23_ladder_isostep_full.yaml \
$SLURM_ARRAY_TASK_ID
