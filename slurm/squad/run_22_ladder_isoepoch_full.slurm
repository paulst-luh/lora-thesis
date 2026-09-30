#!/bin/bash
#SBATCH --job-name=squad-prompting-ladder-isoepoch-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Same reasoning as mnli's ladder-full isoepoch script: 24h x 1.35 ~= 32h.
#SBATCH --time=32:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/squad/prompting_ladder_isoepoch_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_ladder_isoepoch_full/
mkdir -p results/squad/prompting_ladder_isoepoch_full/logs/

python -m src.experiments.runner \
configs/squad/22_ladder_isoepoch_full.yaml \
$SLURM_ARRAY_TASK_ID
