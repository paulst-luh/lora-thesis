#!/bin/bash

sbatch slurm/outlook/run_gemma-4-e4b-it_outlook.slurm
sbatch slurm/outlook/run_gpt-oss-20b_outlook.slurm
sbatch slurm/outlook/run_deepseek-r1-distill-qwen-7b_outlook.slurm
sbatch slurm/outlook/run_llama-3.1-8b-instruct_outlook.slurm
sbatch slurm/outlook/run_mistral-7b-instruct-v0.3_outlook.slurm
