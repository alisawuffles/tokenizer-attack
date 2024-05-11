output_dir=data/mixed_languages_shift
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed
num_languages=10
use_wiki_languages=true

# for t from 89 to 99
for t in {0..99}
do
    if [ ! -d $output_dir/$t ] ; then
        id=$(sbatch \
            --parsable \
            --export=output_dir=$output_dir,num_languages=$num_languages,corpus_dir=$corpus_dir,use_wiki_languages=$use_wiki_languages \
            scripts/train_mixed_tokenizer.sh)
        echo "$t: Submitted batch job $id"
    fi
done
