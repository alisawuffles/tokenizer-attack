output_dir=data/mixed_languages
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed
num_languages=112

# for t from 89 to 99
for t in {0..99}
do
    if [ ! -d $output_dir/$t ] ; then
        id=$(sbatch \
            --parsable \
            --requeue \
            --export=output_dir=$output_dir,num_languages=$num_languages,corpus_dir=$corpus_dir \
            scripts/train_mixed_tokenizer.sh)
        echo "$t: Submitted batch job $id"
    fi
done

# for code
output_dir=data/mixed_code
corpus_dir=/gscratch/scrubbed/alisaliu/redpajama/github/processed
num_languages=10

for t in {0..99}
do
    id=$(sbatch \
        --parsable \
        --requeue \
        --export=output_dir=$output_dir,num_languages=$num_languages,corpus_dir=$corpus_dir \
        scripts/train_mixed_tokenizer.sh)
    echo "$t: Submitted batch job $id"
done
