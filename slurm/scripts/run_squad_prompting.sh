#!/bin/bash

sbatch slurm/squad/run_00_zeroshot.slurm
sbatch slurm/squad/run_01_fft.slurm
sbatch slurm/squad/run_10_main_qv.slurm
sbatch slurm/squad/run_11_main_attention.slurm
sbatch slurm/squad/run_12_main_full.slurm
sbatch slurm/squad/run_20_ladder_isoepoch.slurm
sbatch slurm/squad/run_21_ladder_isostep.slurm
