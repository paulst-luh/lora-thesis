#!/bin/bash

sbatch slurm/gsm8k/run_00_zeroshot.slurm
sbatch slurm/gsm8k/run_01_fft.slurm
sbatch slurm/gsm8k/run_10_main_qv.slurm
sbatch slurm/gsm8k/run_11_main_attention.slurm
sbatch slurm/gsm8k/run_12_main_full.slurm
