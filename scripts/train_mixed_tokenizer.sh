#!/bin/bash
#SBATCH --partition=ckpt
#SBATCH --account=xlab
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=train_tokenizer
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"

echo "Training tokenizers on mixes of $num_languages languages"
python -m train_mixed_tokenizer \
    --output_dir $output_dir \
    --num_languages $num_languages \
    --corpus_dir $corpus_dir \
    ${use_wiki_languages:+--use_wiki_languages "$use_wiki_languages"}
