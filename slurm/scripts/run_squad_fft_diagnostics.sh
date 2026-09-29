#!/bin/bash

sbatch slurm/squad/run_01_fft_wd0.1.slurm
sbatch slurm/squad/run_01_fft_lr_extend.slurm
sbatch slurm/squad/run_01_fft_wd0.1_epochs6.slurm
sbatch slurm/squad/run_01_fft_wd0.1_adamw32bit.slurm
sbatch slurm/squad/run_01_fft_wd0.1_moreseeds.slurm
sbatch slurm/squad/run_01_fft_wd0.1_subset20000.slurm
