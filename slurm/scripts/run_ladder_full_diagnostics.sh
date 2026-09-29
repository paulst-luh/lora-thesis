#!/bin/bash

sbatch slurm/mnli/run_22_ladder_isoepoch_full.slurm
sbatch slurm/mnli/run_23_ladder_isostep_full.slurm
sbatch slurm/squad/run_22_ladder_isoepoch_full.slurm
sbatch slurm/squad/run_23_ladder_isostep_full.slurm
