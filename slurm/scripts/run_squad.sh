#!/bin/bash

sbatch slurm/squad/run_qv.slurm
sbatch slurm/squad/run_attention.slurm
sbatch slurm/squad/run_full.slurm
sbatch slurm/squad/run_fft.slurm