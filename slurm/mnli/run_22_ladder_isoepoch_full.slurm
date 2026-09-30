#!/bin/bash
#SBATCH --job-name=mnli-prompting-ladder-isoepoch-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Same reasoning as the attention ladder's 24h (subset_size up to 50000
# means ~10x the optimizer steps of the main sweep under iso_epoch),
# bumped further since lora.target=full measures ~35% slower per run
# than attention (1183s vs 876s mean, main sweep). 24h x 1.35 ~= 32h.
#SBATCH --time=32:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/mnli/prompting_ladder_isoepoch_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_ladder_isoepoch_full/
mkdir -p results/mnli/prompting_ladder_isoepoch_full/logs/

python -m src.experiments.runner \
configs/mnli/22_ladder_isoepoch_full.yaml \
$SLURM_ARRAY_TASK_ID
