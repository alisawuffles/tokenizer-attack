"""
Count occurrences of tokens (as strings) in a data category. This is used for the baseline based 
on token categorization (TC), which assigns each token to the data category in which it is most frequent.
"""

from pathlib import Path
import click
import os
from tqdm import tqdm
from collections import defaultdict
from utils import read_merges_txt, bytes_to_unicode, truncate_file
from ahocorasick_rs import BytesAhoCorasick
import json

NUM_BYTES = 10**9

oscar_data_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus/processed')
test_dir = Path('data/mixed_languages/n_112')


@click.command()
@click.option(
    '--output_dir',
    type=str,
    help='Looks like data/mixed_languages/n_10/0'
)
@click.option(
    '--corpus_dir',
    type=str,
    help='Looks like /gscratch/scrubbed/alisaliu/oscar-corpus/processed'
)
@click.option(
    '--lang',
    type=str,
)
def main(output_dir: str, corpus_dir: str, lang: str):
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)

    # mapping between individual bytes and unicode chars
    B = bytes_to_unicode()
    Brev = {v: k for k, v in B.items()}

    # construct the vocab in bytes
    merges = read_merges_txt(output_dir / 'merges.txt')
    vocab = [''.join(m.split(' ')) for m in merges]
    vocab = [bytes(Brev[b] for b in v) for v in vocab]

    # build the Aho-Corasick automaton
    ac = BytesAhoCorasick(vocab)

    # read 1G of text files
    # by looping until we have num_bytes or run out of files (do not duplicate!)
    files = []
    byte_count = 0
    lang_files = [f for f in os.listdir(corpus_dir / lang) if 'truncated' not in f]
    for fname in lang_files:
        filesize = os.path.getsize(corpus_dir / lang / fname)
        if byte_count + filesize <= NUM_BYTES:
            files.append(fname)
            byte_count += filesize
        else:
            # truncate to get partial file of the right size
            wanted_filesize = NUM_BYTES - byte_count
            trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
            os.system(f'cp {corpus_dir / lang / fname} {corpus_dir / lang / trunc_fname}')
            trunc_filesize = truncate_file(corpus_dir / lang / trunc_fname, wanted_filesize)

            files.append(trunc_fname)
            byte_count += trunc_filesize

        if byte_count >= NUM_BYTES:
            break

    # loop through files and count the tokens
    chunk_size = 100000
    counts = defaultdict(int)  # elements will look like {1765: 20088}

    for file in files:
        with open(corpus_dir / lang / file, 'rb') as fin:
            text = fin.read()

        overlap = max(len(t) for t in vocab)

        assert chunk_size > overlap, 'chunk_size must be greater than overlap'

        for i in tqdm(range(0, len(text), chunk_size - overlap)):
            chunk = text[i: min(i + chunk_size, len(text))]
            ac_outputs = ac.find_matches_as_indexes(chunk, overlapping=True)
            for vid, start, end in ac_outputs:
                if i == 0 or end > overlap:  # for the 2nd chunk onwards, we need to ignore the overlap
                    counts[vid] += 1

    # convert vocab IDs to byte strings
    counts = {vocab[k]: v for k, v in counts.items()}  # elements will look like {b'\xe5\xb0': 20088}

    # convert byte strings to unicode strings
    counts = {''.join([B[b] for b in token]): count for token, count in counts.items()}  # elements look like {'å°': 40729}

    # save as json
    with open(output_dir / lang / 'token_string_counts.json', 'w') as fout:
        d = {
            'byte_count': byte_count,
            'counts': counts
        }
        json.dump(d, fout, indent=5)


if __name__ == '__main__':
    main()
