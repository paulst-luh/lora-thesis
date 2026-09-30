#!/bin/bash

sbatch slurm/mnli/run_00_zeroshot.slurm
sbatch slurm/mnli/run_01_fft.slurm
sbatch slurm/mnli/run_10_main_qv.slurm
sbatch slurm/mnli/run_11_main_attention.slurm
sbatch slurm/mnli/run_12_main_full.slurm
sbatch slurm/mnli/run_20_ladder_isoepoch.slurm
sbatch slurm/mnli/run_21_ladder_isostep.slurm
