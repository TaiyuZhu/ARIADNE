#!/bin/bash
#SBATCH --job-name=run_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# python -u run_kllm.py --model gpt-oss:20b --prompt Dom_v2
python -u run_kllm_freq.py --model gpt-oss:20b --prompt Alejo_freq