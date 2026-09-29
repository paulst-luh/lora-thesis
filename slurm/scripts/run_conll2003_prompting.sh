#!/bin/bash

sbatch slurm/conll2003/run_00_zeroshot.slurm
sbatch slurm/conll2003/run_01_fft.slurm
sbatch slurm/conll2003/run_10_main_qv.slurm
sbatch slurm/conll2003/run_11_main_attention.slurm
sbatch slurm/conll2003/run_12_main_full.slurm
