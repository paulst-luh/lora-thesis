#!/bin/bash
#SBATCH --job-name=squad-qv

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-53%4

#SBATCH --output=results/squad/qv/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/qv/
mkdir -p results/squad/qv/logs/

python -m src.experiments.runner \
configs/squad/qv.yaml \
$SLURM_ARRAY_TASK_ID