#!/bin/bash
#SBATCH --partition=ckpt
#SBATCH --account=xlab
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=process_oscar
#SBATCH --output="slurm/data/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m process_oscar
