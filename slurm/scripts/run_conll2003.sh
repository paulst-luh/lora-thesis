#!/bin/bash

sbatch slurm/conll2003/run_qv.slurm
sbatch slurm/conll2003/run_attention.slurm
sbatch slurm/conll2003/run_full.slurm
sbatch slurm/conll2003/run_fft.slurm