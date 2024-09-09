# Record all merge frequencies to apply our attack in controlled experiments.
experiment_dir=experiments/mixed_languages/n_112/0
lang_code=en
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed

python -m dump_frequencies \
    --experiment_dir $experiment_dir \
    --lang_code $lang_code \
    --corpus_dir $corpus_dir
