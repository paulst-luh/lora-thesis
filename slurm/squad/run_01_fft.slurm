#!/bin/bash
#SBATCH --job-name=squad-prompting-fft

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/squad/prompting_fft/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_fft/
mkdir -p results/squad/prompting_fft/logs/

python -m src.experiments.runner \
configs/squad/01_fft.yaml \
$SLURM_ARRAY_TASK_ID
