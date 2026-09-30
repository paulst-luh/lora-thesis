#!/bin/bash
#SBATCH --job-name=squad-prompting-fft-wd0.1

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/squad/prompting_fft_wd0.1/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/squad/prompting_fft_wd0.1/
mkdir -p results/squad/prompting_fft_wd0.1/logs/

python -m src.experiments.runner \
configs/squad/01_fft_wd0.1.yaml \
$SLURM_ARRAY_TASK_ID
