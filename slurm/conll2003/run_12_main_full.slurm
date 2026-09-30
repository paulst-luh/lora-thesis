#!/bin/bash
#SBATCH --job-name=conll2003-prompting-main-full

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-71%4

#SBATCH --output=results/conll2003/prompting_main_full/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/conll2003/prompting_main_full/
mkdir -p results/conll2003/prompting_main_full/logs/

python -m src.experiments.runner \
configs/conll2003/12_main_full.yaml \
$SLURM_ARRAY_TASK_ID
