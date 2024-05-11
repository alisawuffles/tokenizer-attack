big_experiment_dir=data/mixed_languages
corpus_dir=/gscratch/scrubbed/alisaliu/oscar-corpus/processed
num_languages=10

for t in {0..99}
do
    experiment_dir=$big_experiment_dir/n_$num_languages/$t
    for lang_code in $(ls $experiment_dir)
    do
        # if $experiment_dir/$lang_code is a directory and all_pair_counts is not already inside it
        if [ -d $experiment_dir/$lang_code ] && [ ! -f $experiment_dir/$lang_code/all_pair_counts.json ] ; then
            id=$(sbatch \
                --parsable \
                --requeue \
                --export=experiment_dir=$experiment_dir,lang_code=$lang_code,corpus_dir=$corpus_dir,model_name=$model_name \
                scripts/dump_freqs.sh)
            echo "n_$num_languages/$t/$lang_code: Submitted batch job $id"
        fi
    done
done


big_experiment_dir=data/mixed_code
corpus_dir=/gscratch/scrubbed/alisaliu/redpajama/github/processed
num_languages=10

for t in {0..3}
do
    experiment_dir=$big_experiment_dir/n_$num_languages/$t
    for lang_code in $(ls $experiment_dir)
    do
        # if $experiment_dir/$lang_code is a directory and all_pair_counts is not already inside it
        if [ -d $experiment_dir/$lang_code ] && [ ! -f $experiment_dir/$lang_code/all_pair_counts.json ] ; then
            id=$(sbatch \
                --parsable \
                --requeue \
                --export=experiment_dir=$experiment_dir,lang_code=$lang_code,corpus_dir=$corpus_dir,model_name=$model_name \
                scripts/dump_freqs.sh)
            echo "n_$num_languages/$t/$lang_code: Submitted batch job $id"
        fi
    done
done

big_experiment_dir=data/mixed_language_shift
corpus_dir=/gscratch/scrubbed/alisaliu/redpajama/wikipedia/processed
num_languages=10

for t in {0..93}
do
    experiment_dir=$big_experiment_dir/n_$num_languages/$t
    for lang_code in $(ls $experiment_dir)
    do
        # if $experiment_dir/$lang_code is a directory and all_pair_counts is not already inside it
        if [ -d $experiment_dir/$lang_code ] && [ ! -f $experiment_dir/$lang_code/all_pair_counts.json ] ; then
            id=$(sbatch \
                --parsable \
                --requeue \
                --export=experiment_dir=$experiment_dir,lang_code=$lang_code,corpus_dir=$corpus_dir,model_name=$model_name \
                scripts/dump_freqs.sh)
            echo "n_$num_languages/$t/$lang_code: Submitted batch job $id"
        fi
    done
done
