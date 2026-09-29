#!/bin/bash

sbatch slurm/mnli/run_qv.slurm
sbatch slurm/mnli/run_attention.slurm
sbatch slurm/mnli/run_full.slurm
sbatch slurm/mnli/run_fft.slurm