# Train tokenizer on a randomly sampled mix of n data categories, saving the output in output_dir.
n=112
test_id=0
output_dir=experiments/mixed_languages/n_${n}/$test_id
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed

echo "Training tokenizers on mixes of $num_categories languages"
python -m train_mixed_tokenizer \
    --output_dir $output_dir \
    --num_categories $n \
    --corpus_dir $corpus_dir
