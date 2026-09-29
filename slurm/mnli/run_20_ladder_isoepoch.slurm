#!/bin/bash
#SBATCH --job-name=mnli-prompting-ladder-isoepoch

#SBATCH --partition=kisski
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#
# Longer than the 8h used by every other block here on purpose: this
# is the one config whose per-task compute isn't bounded by the
# existing 5000-example precedent -- dataset.subset_size goes up to
# 50000, and under iso_epoch (fixed epoch count) that rung runs ~10x
# the optimizer steps of the 5000-example runs elsewhere in this study
# (~9375 steps vs ~938). This estimate is extrapolated, not measured --
# watch the first few 50000-subset array tasks and adjust if they
# under- or over-run this budget.
#

#SBATCH --time=24:00:00

#SBATCH --array=0-119%4

#SBATCH --output=results/mnli/prompting_ladder_isoepoch/logs/%x_%A_%a.out

source ~/lora-thesis/lora-env/bin/activate

cd ~/lora-thesis

mkdir -p results/mnli/prompting_ladder_isoepoch/
mkdir -p results/mnli/prompting_ladder_isoepoch/logs/

python -m src.experiments.runner \
configs/mnli/20_ladder_isoepoch.yaml \
$SLURM_ARRAY_TASK_ID
