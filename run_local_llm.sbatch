#!/bin/bash
#SBATCH --job-name=run_llm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G            
#SBATCH --gres=gpu:1   
# SBATCH --partition=slam_gpu                      
# SBATCH --constraint=l40s                  

python run_local_llm.py