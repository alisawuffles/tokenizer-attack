#!/bin/bash
#SBATCH --partition=ark
#SBATCH --account=ckpt
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_tokenizer
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m train_tokenizer
