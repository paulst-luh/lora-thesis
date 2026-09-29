#!/bin/bash

sbatch slurm/gsm8k/run_qv.slurm
sbatch slurm/gsm8k/run_attention.slurm
sbatch slurm/gsm8k/run_full.slurm
sbatch slurm/gsm8k/run_fft.slurm