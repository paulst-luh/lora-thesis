#!/bin/bash
#SBATCH --job-name=conll2003-prompting-fft-subset12000

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ~2.4x the data at the same 3 epochs -> roughly 2.4x the training
# time. Measured baseline here is ~909s, so ~2180s worst case -- 8h
# leaves ample margin.
#SBATCH --time=08:00:00

#SBATCH --array=0-8%4

#SBATCH --output=results/conll2003/prompting_fft_wd0.1_subset12000/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/conll2003/prompting_fft_wd0.1_subset12000/
mkdir -p results/conll2003/prompting_fft_wd0.1_subset12000/logs/

python -m src.experiments.runner \
configs/conll2003/01_fft_wd0.1_subset12000.yaml \
$SLURM_ARRAY_TASK_ID
