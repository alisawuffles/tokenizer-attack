"""
This script will apply a tokenizer's merges one by one to a data corpus, and record the frequencies
of all possible merges at each time step. Note that only merge counts that *differ* from the previous
time step are recorded, to save space.
"""

import os
from pathlib import Path
import random
import time
from utils import ensure_dir, train_tokenizer_or_dump_frequencies, truncate_file, read_json
import click
import json

DEFAULT_NUM_BYTES = 10**9


@click.command()
@click.option(
    '--experiment_dir',
    type=str,
    default='data/mixed_languages/n_10/0'
)
@click.option(
    '--lang_code',
    type=str,
    default='en',
    help='The language to dump frequencies for.'
)
@click.option(
    '--corpus_dir',
    type=str,
    default='/gscratch/scrubbed/alisaliu/oscar-corpus/processed',
    help='Directory containing language subdirectories with text files, to use for estimating merge frequencies.'
)
@click.option(
    '--model_name',
    type=str,
    default=None,
    help='Use this when applying merges from a commercial tokenizer.'
)
@click.option(
    '--num_bytes',
    type=int,
    default=DEFAULT_NUM_BYTES
)
def main(experiment_dir: str, lang_code: str, corpus_dir: str, model_name: str, num_bytes: int):
    corpus_dir = Path(corpus_dir)
    experiment_dir = Path(experiment_dir)
    lang_dir = lang_code if num_bytes == DEFAULT_NUM_BYTES else f'{lang_code}/{"{:.0e}".format(num_bytes).replace("e+", "e")}'
    os.chdir('/gscratch/xlab/alisaliu/hack-tokenizers')

    print(f'We will dump frequencies in {experiment_dir}/{lang_dir}...')

    # cd into the folder because tokenizer.train() will check to see if merges.txt exists here
    os.chdir(experiment_dir)

    # get text data
    if num_bytes != DEFAULT_NUM_BYTES:
        print(f'We will dump frequencies using {num_bytes} bytes of text data.', flush=True)

    all_text_files = [str(corpus_dir / lang_code / f) for f in os.listdir(corpus_dir / lang_code) if f.endswith('txt') and 'truncated' not in f]
    random.shuffle(all_text_files)
    byte_count = 0
    text_files = []

    # keep reading text files until we have num_bytes or run out of files (do not duplicate!)
    while byte_count < num_bytes and all_text_files:
        fname = all_text_files.pop()
        filesize = os.path.getsize(corpus_dir / lang_code / fname)
        if byte_count + filesize <= num_bytes:
            text_files.append(str(corpus_dir / lang_code / fname))
            byte_count += filesize
        else:
            wanted_filesize = num_bytes - byte_count
            trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
            os.system(f'cp {corpus_dir / lang_code / fname} {corpus_dir / lang_code / trunc_fname}')
            truncate_file(corpus_dir / lang_code / trunc_fname, wanted_filesize)
            text_files.append(str(corpus_dir / lang_code / trunc_fname))
            byte_count += wanted_filesize

    print(f'Loaded {len(text_files)} text files!', flush=True)

    print('Training tokenizer...', flush=True)
    start_time = time.time()
    tokenizer = train_tokenizer_or_dump_frequencies(text_files, model_name=model_name)
    print(f'Train time: {time.time() - start_time}', flush=True)

    ensure_dir(lang_dir)
    tokenizer.model.save(lang_dir)

    num_pairs = sum(read_json(f'{lang_dir}/all_pair_counts.json')[0].values())

    with open(f'{lang_dir}/meta.json', 'w') as fo:
        config = {}
        config['byte_count'] = byte_count
        config['text_files'] = text_files
        config['pairs'] = num_pairs
        json.dump(config, fo, indent=5)

    # delete merges.txt and vocab.json because we don't need it
    os.remove(f'{lang_dir}/merges.txt')
    os.remove(f'{lang_dir}/vocab.json')
    print('Tokenizer files saved to ' + str(experiment_dir / lang_dir), flush=True)

    # Delete files that were constructed just for this
    for f in text_files:
        if os.path.exists(f) and 'truncated' in f:
            os.remove(f)


if __name__ == '__main__':
    main()
