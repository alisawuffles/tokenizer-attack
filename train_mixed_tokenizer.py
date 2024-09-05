"""
Train tokenizer on a mixture of data categories. The categories and their weights are sampled from the unit simplex.
"""

import os
from pathlib import Path
import time
import re
import json
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import click
from utils import ensure_dir, truncate_file


def sample_from_unit_simplex(n, M=10000):
    """
    Smith & Trombe algorithm as described in https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    """
    x = random.sample(range(1, M), n - 1)
    x = [0] + sorted(x) + [M]
    new_x = []
    for i in range(1, len(x)):
        new_x.append((x[i] - x[i - 1]) / M)
    return new_x


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
    '--num_categories',
    type=int,
    default=10,
    help='Number of languages to include in the mixed tokenizer training set. Weights will be sampled.'
)
@click.option(
    '--total_bytes',
    type=int,
    default=10**10,
    help='The maximum number of bytes to use for tokenizer training.'
)
@click.option(
    '--corpus_dir',
    type=str,
    default='/gscratch/scrubbed/alisaliu/oscar-corpus/processed',
    help='Directory containing subdirectories for each category, to use for training the tokenizer.'
)
@click.option(
    '--use_wiki_languages',
    type=bool,
    default=False,
    help="Whether to use only the set of languages available in RedPajama's wikipedia split. This is used for the distribution shift experiment."
)
@click.option(
    '--size_threshold',
    type=int,
    default=None,
    help='If specified, we will only use languages with byte counts above this threshold. This is used for the data size ablation.'
)
def main(
    output_dir: str,
    use_spm: bool,
    num_categories: int,
    total_bytes: int,
    corpus_dir: str,
    use_wiki_languages: bool,
    size_threshold: int
):
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f'We are training a tokenizer for {output_dir}', flush=True)

    # If meta.json exists, use the mixture and files from there.
    if os.path.exists(output_dir / 'meta.json'):
        print('Output directory contains meta.json, so we will use the languages and weights from there.', flush=True)
        meta = json.load(open(output_dir / 'meta.json'))
        byte_counts = meta['byte_count']
        languages = byte_counts.keys()
        weights = [v / sum(byte_counts.values()) for v in byte_counts.values()]
        train_files = meta['train_files']
        text_files = []
        for file, count in train_files.items():
            # file = file.replace('/gscratch/scrubbed/alisaliu/redpajama/github', '/gscratch/xlab/alisaliu/redpajama/github')
            if not os.path.exists(file):
                assert 'truncated' in file, f'{file} not found'
                untruncated_file = re.sub(r'_truncated_\d+', '', file)
                wanted_filesize = int(re.search(r'_truncated_(\d+)', file).group(1))
                os.system(f'cp {untruncated_file} {file}')
                truncate_file(file, wanted_filesize)
            text_files.extend([file] * count)

    # Else, sample languages and weights.
    else:
        if use_wiki_languages:
            print('Using only languages available in RedPajama wikipedia split.', flush=True)
            languages = random.sample(os.listdir('/gscratch/xlab/alisaliu/redpajama/wikipedia'), num_categories)
        elif size_threshold:
            from constants import keyword_to_mixture_type
            for keyword in keyword_to_mixture_type:
                if keyword in str(output_dir):
                    CATEGORY_SIZES = keyword_to_mixture_type[keyword]

            all_languages = [lang for lang, size in CATEGORY_SIZES.items() if size > size_threshold]
            languages = random.sample(all_languages, num_categories)
            print(f'Using the {len(all_languages)} languages with at least {size_threshold} bytes of text.', flush=True)
        else:
            languages = random.sample(os.listdir(corpus_dir), num_categories)

        weights = sample_from_unit_simplex(len(languages))
        print(f'Intended language distribution: {({l: w for l, w in zip(languages, weights)})}', flush=True)

        tqdm_bar = tqdm(total=total_bytes, desc='Loading text data')
        text_files, byte_counts = defaultdict(list), defaultdict(int)

        # Get text data for each language, duplicating as necessary until byte_counts for each language is met.
        for lang_code, weight in zip(languages, weights):
            language_files = [f for f in os.listdir(corpus_dir / lang_code) if f.endswith('.txt') and 'truncated' not in f]
            random.shuffle(language_files)
            counter = 0
            while byte_counts[lang_code] < int(weight * total_bytes):
                fname = language_files[counter % len(language_files)]
                filesize = os.path.getsize(corpus_dir / lang_code / fname)
                if byte_counts[lang_code] + filesize <= weight * total_bytes:
                    text_files[lang_code].append(str(corpus_dir / lang_code / fname))
                    byte_counts[lang_code] += filesize
                    tqdm_bar.update(filesize)
                else:
                    # make a copy, then truncate file to desired size
                    wanted_filesize = int(weight * total_bytes) - byte_counts[lang_code]
                    trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
                    os.system(f'cp {corpus_dir / lang_code / fname} {corpus_dir / lang_code / trunc_fname}')
                    truncate_file(corpus_dir / lang_code / trunc_fname, wanted_filesize)
                    text_files[lang_code].append(str(corpus_dir / lang_code / trunc_fname))
                    byte_counts[lang_code] += wanted_filesize
                    tqdm_bar.update(wanted_filesize)
                counter += 1

        # Write metadata.
        text_files = [f for lang_files in text_files.values() for f in lang_files]
        with open(output_dir / 'meta.json', 'w') as fo:
            meta = {}
            meta['byte_count'] = byte_counts
            meta['total_bytes'] = total_bytes
            meta['train_files'] = Counter(text_files)
            json.dump(meta, fo, indent=5)

    print(f'Real language distribution: {byte_counts}', flush=True)

    # Train tokenizer
    start_time = time.time()

    if not use_spm:
        from utils import train_tokenizer_or_dump_frequencies
        print('Training with HF tokenizers...')
        tokenizer = train_tokenizer_or_dump_frequencies(text_files)
        tokenizer.model.save(str(output_dir))
        tokenizer.save(str(output_dir / 'tokenizer.json'))
    else:
        from utils import train_tokenizer_spm
        print('Training with SentencePiece...')
        train_tokenizer_spm(text_files, output_dir)

    print(f'Train time: {time.time() - start_time}', flush=True)

    # Create an empty directory for each language (this will be helpful later)
    for lang_code in languages:
        ensure_dir(output_dir / lang_code)

    print('Tokenizer info saved to ' + str(output_dir), flush=True)

    # Delete files that were constructed just for this
    for f in text_files:
        if os.path.exists(f) and 'truncated' in f:
            os.remove(f)


if __name__ == '__main__':
    main()
