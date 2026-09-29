#!/bin/bash
#SBATCH --job-name=squad-prompting-zeroshot

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=01:00:00

#SBATCH --array=0-2

#SBATCH --output=results/squad/prompting_zeroshot/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_zeroshot/
mkdir -p results/squad/prompting_zeroshot/logs/

python -m src.experiments.runner \
configs/squad/00_zeroshot.yaml \
$SLURM_ARRAY_TASK_ID
