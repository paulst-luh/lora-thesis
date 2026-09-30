#!/bin/bash
#SBATCH --job-name=squad-prompting-ladder-isostep-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Same reasoning as mnli's ladder-full isostep script: 8h x 1.35 ~= 11h,
# rounded up to 12h.
#SBATCH --time=12:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/squad/prompting_ladder_isostep_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_ladder_isostep_full/
mkdir -p results/squad/prompting_ladder_isostep_full/logs/

python -m src.experiments.runner \
configs/squad/23_ladder_isostep_full.yaml \
$SLURM_ARRAY_TASK_ID
