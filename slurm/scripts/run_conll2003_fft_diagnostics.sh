#!/bin/bash

sbatch slurm/conll2003/run_01_fft_wd0.1.slurm
sbatch slurm/conll2003/run_01_fft_lr_extend.slurm
sbatch slurm/conll2003/run_01_fft_wd0.1_epochs6.slurm
sbatch slurm/conll2003/run_01_fft_wd0.1_adamw32bit.slurm
sbatch slurm/conll2003/run_01_fft_wd0.1_moreseeds.slurm
sbatch slurm/conll2003/run_01_fft_wd0.1_subset12000.slurm
