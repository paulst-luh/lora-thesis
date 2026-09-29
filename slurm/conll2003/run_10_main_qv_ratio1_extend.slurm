#!/bin/bash
#SBATCH --job-name=conll2003-prompting-main-qv-ratio1-extend

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=08:00:00

#SBATCH --array=0-17%4

#SBATCH --output=results/conll2003/prompting_main_qv_ratio1_extend/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/conll2003/prompting_main_qv_ratio1_extend/
mkdir -p results/conll2003/prompting_main_qv_ratio1_extend/logs/

python -m src.experiments.runner \
configs/conll2003/10_main_qv_ratio1_extend.yaml \
$SLURM_ARRAY_TASK_ID
