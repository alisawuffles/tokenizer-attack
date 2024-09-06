"""
Train tokenizer on a single data category.
"""

import os
from pathlib import Path
import time
import json
import random
from tqdm import tqdm
import click
from utils import ensure_dir, truncate_file


@click.command()
@click.option(
    '--output_dir',
    type=str,
    help='Where to save trained tokenizers, e.g., data/mixed_languages/n_10/0.'
)
@click.option(
    '--use_spm',
    type=bool,
    default=False,
    help='Whether to use SentencePiece library for training. Default is tokenizers.'
)
@click.option(
    '--total_bytes',
    type=int,
    default=10**10,
    help='The maximum number of characters to use for tokenizer training.'
)
@click.option(
    '--corpus_dir',
    type=str,
    default='/gscratch/scrubbed/alisaliu/oscar-corpus/processed/en',
    help='Directory containing text files to use for training the tokenizer.'
)
def main(
    output_dir: str,
    use_spm: bool,
    total_bytes: int,
    corpus_dir: str,
):
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f'We are training a tokenizer for {output_dir}', flush=True)

    tqdm_bar = tqdm(total=total_bytes, desc='Loading text data')
    train_files, byte_count = [], 0

    # get text data for each language, duplicating as necessary until byte_counts for each language is met
    language_files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt') and 'truncated' not in f]
    random.shuffle(language_files)
    counter = 0
    while byte_count < total_bytes:
        fname = language_files[counter % len(language_files)]
        filesize = os.path.getsize(corpus_dir / fname)
        if byte_count + filesize <= total_bytes:
            train_files.append(str(corpus_dir / fname))
            byte_count += filesize
            tqdm_bar.update(filesize)
        else:
            # make a copy, then truncate file to desired size
            wanted_filesize = int(total_bytes) - byte_count
            trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
            os.system(f'cp {corpus_dir / fname} {corpus_dir / trunc_fname}')
            truncate_file(corpus_dir / trunc_fname, wanted_filesize)
            train_files.append(str(corpus_dir / trunc_fname))
            byte_count += wanted_filesize
            tqdm_bar.update(wanted_filesize)
        counter += 1

    # Write metadata
    with open(output_dir / 'meta.json', 'w') as fo:
        meta = {}
        meta['total_bytes'] = total_bytes
        meta['train_files'] = train_files
        json.dump(meta, fo, indent=5)

    # Train tokenizer
    start_time = time.time()

    if not use_spm:
        from utils import train_tokenizer_or_dump_frequencies
        print('Training with HF tokenizers...')
        tokenizer = train_tokenizer_or_dump_frequencies(train_files)
        tokenizer.model.save(str(output_dir))
        tokenizer.save(str(output_dir / 'tokenizer.json'))
    else:
        from utils import train_tokenizer_spm
        print('Training with SentencePiece...')
        train_tokenizer_spm(train_files, output_dir)

    print(f'Train time: {time.time() - start_time}', flush=True)
    print('Tokenizer info saved to ' + str(output_dir), flush=True)

    # Delete files that were constructed just for this
    for f in train_files:
        if 'truncated' in f:
            os.remove(f)


if __name__ == '__main__':
    main()
