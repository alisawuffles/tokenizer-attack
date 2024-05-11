# train tokenizer from scratch on small amount of data
python -m train_mixed_tokenizer \
    --output_dir data/test_mem_usage \
    --num_languages 1 \
    --total_chars 1000000000 \
    --corpus_dir /gscratch/scrubbed/alisaliu/oscar-corpus/processed

# apply the tokenizer to the same text as used for training
python -m dump_frequencies \
    --experiment_dir data/test_mem_usage/n_1/2 \
    --lang_code pa \
    --max_files 1 \
    --corpus_dir /gscratch/scrubbed/alisaliu/oscar-corpus/processed
