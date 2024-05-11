experiment_dir=data/llm_tokenizers/llama3
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed
# corpus_dir=/gscratch/scrubbed/alisaliu/redpajama/github/processed
model_name=llama3

for lang_code in $(ls $corpus_dir)
do
    # if all_pair_counts is not already there (don't overwrite stuff!)
    if [ ! -f $experiment_dir/$lang_code/all_pair_counts.json ] ; then
        id=$(sbatch \
            --parsable \
            --requeue \
            --export=experiment_dir=$experiment_dir,lang_code=$lang_code,corpus_dir=$corpus_dir,model_name=$model_name \
            scripts/dump_freqs.sh
        )
        echo "$lang_code: Submitted batch job $id"
        # echo $lang_code
    fi
done
