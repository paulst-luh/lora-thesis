#!/bin/bash
#SBATCH --job-name=squad-prompting-ladder-isostep-r64-lr5e4

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-11%4

#SBATCH --output=results/squad/prompting_ladder_isostep_r64_lr5e4/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_ladder_isostep_r64_lr5e4/
mkdir -p results/squad/prompting_ladder_isostep_r64_lr5e4/logs/

python -m src.experiments.runner \
configs/squad/21_ladder_isostep_r64_lr5e4.yaml \
$SLURM_ARRAY_TASK_ID
