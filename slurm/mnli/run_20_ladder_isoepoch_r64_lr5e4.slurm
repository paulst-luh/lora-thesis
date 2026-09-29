#!/bin/bash
#SBATCH --job-name=mnli-prompting-ladder-isoepoch-r64-lr5e4

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=24:00:00

#SBATCH --array=0-11%4

#SBATCH --output=results/mnli/prompting_ladder_isoepoch_r64_lr5e4/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_ladder_isoepoch_r64_lr5e4/
mkdir -p results/mnli/prompting_ladder_isoepoch_r64_lr5e4/logs/

python -m src.experiments.runner \
configs/mnli/20_ladder_isoepoch_r64_lr5e4.yaml \
$SLURM_ARRAY_TASK_ID
