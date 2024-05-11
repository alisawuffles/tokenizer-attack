import os
from pathlib import Path
import random
import time
from utils import ensure_dir, train_tokenizer_or_dump_frequencies
import click
import json


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
    default=10**9
)
def main(experiment_dir: str, lang_code: str, corpus_dir: str, model_name: str, num_bytes: int):
    corpus_dir = Path(corpus_dir)
    experiment_dir = Path(experiment_dir)
    os.chdir('/gscratch/xlab/alisaliu/hack-tokenizers')

    print(f'We will dump frequencies in {experiment_dir}/{lang_code}...')

    # cd into the folder because tokenizer.train() will check to see if merges.txt exists here
    os.chdir(experiment_dir)
    print('Current directory:', os.getcwd(), flush=True)

    print('Initializing tokenizer...', flush=True)

    # get text data
    all_text_files = [str(corpus_dir / lang_code / f) for f in os.listdir(corpus_dir / lang_code) if f.endswith('txt') ]
    random.shuffle(all_text_files)
    byte_count = 0
    text_files = []
    # keep reading text files until we have num_bytes
    while byte_count < num_bytes:
        fname = all_text_files.pop()
        filesize = os.path.getsize(corpus_dir / lang_code / fname)
        if filesize < num_bytes:
            text_files.append(str(corpus_dir / lang_code / fname))
            byte_count += filesize
        else:
            wanted_filesize = num_bytes - byte_count
            trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
            os.system(f'cp {corpus_dir / lang_code / fname} {corpus_dir / lang_code / trunc_fname}')
            with open(corpus_dir / lang_code / trunc_fname, 'a') as fin:
                fin.truncate(wanted_filesize)
            text_files.append(str(corpus_dir / lang_code / trunc_fname))
            byte_count += wanted_filesize
    print(f'Loaded {len(text_files)} text files!', flush=True)

    ensure_dir(lang_code)
    with open(f'{lang_code}/meta.json', 'w') as fo:
        config = {}
        config['byte_count'] = byte_count
        config['text_files'] = text_files
        json.dump(config, fo, indent=5)

    print('Training tokenizer...', flush=True)
    start_time = time.time()
    tokenizer = train_tokenizer_or_dump_frequencies(text_files, model_name=model_name)
    print(f'Train time: {time.time() - start_time}', flush=True)
    tokenizer.model.save(lang_code)

    # delete merges.txt and vocab.json because we don't need it
    os.remove(f'{lang_code}/merges.txt')
    os.remove(f'{lang_code}/vocab.json')
    print('Tokenizer files saved to ' + str(experiment_dir / lang_code), flush=True)


if __name__ == '__main__':
    main()
