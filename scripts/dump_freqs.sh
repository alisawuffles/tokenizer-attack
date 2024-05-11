#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --job-name=dump_frequency
#SBATCH --output="slurm/dump/slurm-%J-%x.out"
#SBATCH --error="slurm/error/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m dump_frequencies \
    --experiment_dir "$experiment_dir" \
    --lang_code "$lang_code" \
    --corpus_dir "$corpus_dir" \
    ${model_name:+--model_name "$model_name"}
