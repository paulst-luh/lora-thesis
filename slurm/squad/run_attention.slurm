#!/bin/bash
#SBATCH --job-name=squad-attention

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-53%4

#SBATCH --output=results/squad/attention/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/attention/
mkdir -p results/squad/attention/logs/

python -m src.experiments.runner \
configs/squad/attention.yaml \
$SLURM_ARRAY_TASK_ID