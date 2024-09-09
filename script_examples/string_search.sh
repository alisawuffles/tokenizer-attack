# Search for each token *as a string* in the given corpus. 
# This is for the baseline based on token categorization (TC).
test_id=0
experiment_dir=experiments/mixed_languages/n_112/$test_id
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed
lang_code=en

python -m search_string \
    --output_dir $output_dir \
    --corpus_dir $corpus_dir \
    --lang $lang_code
