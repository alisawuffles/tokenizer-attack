import os
from pathlib import Path
import time
from utils import ensure_dir
import json
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import click
from utils import train_tokenizer_or_dump_frequencies


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


def is_valid_unicode(data):
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def truncate_file(filename, wanted_filesize):
    with open(filename, 'rb') as f:
        f.seek(wanted_filesize)
        data = f.read(1)
        while data and is_valid_unicode(data):
            data = f.read(1)
            wanted_filesize += 1
    with open(filename, 'r+', encoding='utf-8') as fin:
        fin.truncate(wanted_filesize)


@click.command()
@click.option(
    '--output_dir',
    type=str,
    default='data/mixed_languages',
    help='Where to save trained tokenizers.'
)
@click.option(
    '--num_languages',
    type=int,
    default=40,
    help='Number of languages to include in the mixed tokenizer training set. Weights will be sampled.'
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
    default='/gscratch/scrubbed/alisaliu/oscar-corpus/processed',
    help='Directory containing language subdirectories with text files, to use for training the tokenizer.'
)
@click.option(
    '--use_wiki_languages',
    type=bool,
    default=False,
    help="Whether to use only the set of languages available in RedPajama's wikipedia split."
)
def main(
    output_dir: str,
    num_languages: int,
    total_bytes: int,
    corpus_dir: str,
    use_wiki_languages: bool
):
    corpus_dir = Path(corpus_dir)
    trained_tokenizers_dir = Path(f'{output_dir}/n_{num_languages}')
    ensure_dir(trained_tokenizers_dir)

    if not use_wiki_languages:
        languages = random.sample(os.listdir(corpus_dir), num_languages)
    else:
        print('Using only languages available in RedPajama wikipedia split.', flush=True)
        languages = random.sample(os.listdir('/gscratch/scrubbed/alisaliu/redpajama/wikipedia/processed'), num_languages)

    weights = sample_from_unit_simplex(len(languages))
    print(f'Intended language distribution: {({l: w for l, w in zip(languages, weights)})}', flush=True)

    tqdm_bar = tqdm(total=total_bytes, desc='Loading text data')
    text_files, byte_counts = defaultdict(list), defaultdict(int)

    # get text data for each language, duplicating as necessary until byte_counts for each language is met
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
                # with open(corpus_dir / lang_code / trunc_fname, 'a') as fin:
                #     fin.truncate(wanted_filesize)
                text_files[lang_code].append(str(corpus_dir / lang_code / trunc_fname))
                byte_counts[lang_code] += wanted_filesize
                tqdm_bar.update(wanted_filesize)
            counter += 1

    print(f'Real language distribution: {byte_counts}', flush=True)

    print('Training tokenizer...', flush=True)
    text_files = [f for lang_files in text_files.values() for f in lang_files]
    start_time = time.time()
    tokenizer = train_tokenizer_or_dump_frequencies(text_files)
    print(f'Train time: {time.time() - start_time}', flush=True)

    # write outputs!
    dirname = str(len(os.listdir(trained_tokenizers_dir)))
    ensure_dir(trained_tokenizers_dir / dirname)
    tokenizer.model.save(str(trained_tokenizers_dir / dirname))

    with open(trained_tokenizers_dir / dirname / 'meta.json', 'w') as fo:
        config = {}
        config['byte_count'] = byte_counts
        config['total_bytes'] = total_bytes
        config['train_files'] = Counter(text_files)
        json.dump(config, fo, indent=5)

    # create an empty directory for each language (this will be helpful later)
    for lang_code in languages:
        ensure_dir(trained_tokenizers_dir / dirname / lang_code)

    print('Tokenizer info saved to ' + str(trained_tokenizers_dir / dirname), flush=True)

    # delete files that were constructed just for this
    for f in text_files:
        if 'truncated' in f:
            os.remove(f)


if __name__ == '__main__':
    main()
