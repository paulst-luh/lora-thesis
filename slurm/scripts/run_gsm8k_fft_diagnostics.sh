#!/bin/bash

sbatch slurm/gsm8k/run_01_fft_wd0.1.slurm
sbatch slurm/gsm8k/run_01_fft_lr_extend.slurm
sbatch slurm/gsm8k/run_01_fft_wd0.1_epochs6.slurm
sbatch slurm/gsm8k/run_01_fft_wd0.1_adamw32bit.slurm
sbatch slurm/gsm8k/run_01_fft_wd0.1_moreseeds.slurm
sbatch slurm/gsm8k/run_01_fft_wd0.1_subset7000.slurm
