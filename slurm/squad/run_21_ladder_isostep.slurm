#!/bin/bash
#SBATCH --job-name=squad-prompting-ladder-isostep

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#
# iso_step fixes max_steps=938 at every subset_size rung (see
# configs/squad/21_ladder_isostep.yaml), so per-task compute here
# matches the 8h precedent used by the main sweep, unlike its
# isoepoch sibling.
#

#SBATCH --time=08:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/squad/prompting_ladder_isostep/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_ladder_isostep/
mkdir -p results/squad/prompting_ladder_isostep/logs/

python -m src.experiments.runner \
configs/squad/21_ladder_isostep.yaml \
$SLURM_ARRAY_TASK_ID
