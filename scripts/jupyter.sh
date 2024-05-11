#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --output="slurm/notebook/slurm-%J.out"

jupyter notebook --no-browser --ip 0.0.0.0 --port=8008
