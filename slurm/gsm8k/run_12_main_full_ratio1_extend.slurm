#!/bin/bash
#SBATCH --job-name=gsm8k-prompting-main-full-ratio1-extend

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-17%4

#SBATCH --output=results/gsm8k/prompting_main_full_ratio1_extend/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/gsm8k/prompting_main_full_ratio1_extend/
mkdir -p results/gsm8k/prompting_main_full_ratio1_extend/logs/

python -m src.experiments.runner \
configs/gsm8k/12_main_full_ratio1_extend.yaml \
$SLURM_ARRAY_TASK_ID
