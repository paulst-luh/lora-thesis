#!/bin/bash

#
# Submits ONLY the ratio-1 grid completion -- (r=4, alpha=4) and
# (r=64, alpha=64) x 3 learning rates x 3 seeds, all three targets, all
# four datasets (configs/<dataset>/1*_main_*_ratio1_extend.yaml):
# 12 array jobs x 18 runs = 216 runs. Nothing from the original main
# sweep is resubmitted.
#
# The logs/ directories are created up front: SLURM does not create the
# --output directory itself, and opens that file before the job
# script's own mkdir ever runs.
#

cd ~/lora-thesis

mkdir -p results/{mnli,squad,conll2003,gsm8k}/prompting_main_{qv,attention,full}_ratio1_extend/logs

sbatch slurm/mnli/run_10_main_qv_ratio1_extend.slurm
sbatch slurm/mnli/run_11_main_attention_ratio1_extend.slurm
sbatch slurm/mnli/run_12_main_full_ratio1_extend.slurm

sbatch slurm/squad/run_10_main_qv_ratio1_extend.slurm
sbatch slurm/squad/run_11_main_attention_ratio1_extend.slurm
sbatch slurm/squad/run_12_main_full_ratio1_extend.slurm

sbatch slurm/conll2003/run_10_main_qv_ratio1_extend.slurm
sbatch slurm/conll2003/run_11_main_attention_ratio1_extend.slurm
sbatch slurm/conll2003/run_12_main_full_ratio1_extend.slurm

sbatch slurm/gsm8k/run_10_main_qv_ratio1_extend.slurm
sbatch slurm/gsm8k/run_11_main_attention_ratio1_extend.slurm
sbatch slurm/gsm8k/run_12_main_full_ratio1_extend.slurm
