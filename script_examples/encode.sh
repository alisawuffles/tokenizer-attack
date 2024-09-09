# Count the number of tokens required to encode the given language corpus. 
# This is used for the baseline based on tokenizer encoding efficiency (TEE).
test_id=0
experiment_dir=experiments/mixed_languages/n_112/$test_id
tokenizer_path=$experiment_dir/tokenizer.json
lang_code=en
output_dir=$experiment_dir/$lang_code
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed

echo "Encode data using tokenizer at $tokenizer_path"
python -m encode_category_data \
    --tokenizer_path $tokenizer_path \
    --lang $lang_code \
    --output_dir $output_dir \
    --corpus_dir $corpus_dir


# You'll also have to do this for a single-language tokenizer.

