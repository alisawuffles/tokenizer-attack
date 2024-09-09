"""
Count the number of tokens required to encode a language corpus. This is used for calculating the byte-to-token ratio
for the baseline based on tokenizer encoding efficiency.
"""
import json
from pathlib import Path
from tokenizers import Tokenizer
import click
import random
import os
from tqdm import tqdm
from utils import truncate_file

NUM_BYTES = 10**9


@click.command()
@click.option(
    '--tokenizer_path',
    type=str,
    help='Looks like data/mixed_languages/n_10/0/tokenizer.json'
)
@click.option(
    '--lang',
    type=str,
)
@click.option(
    '--output_dir',
    type=str,
)
@click.option(
    '--corpus_dir',
    type=str
)
def main(tokenizer_path: str, lang: str, output_dir: str, corpus_dir: str):
    output_dir = Path(output_dir)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    corpus_dir = Path(corpus_dir)

    def count_tokens(file):
        """
        Encode file and return the number of tokens.
        """
        num_tokens = 0

        with open(file) as fin:
            text = fin.read()
            # split into chunks so we don't OOM
            pps = text.split('\n\n')
            chunk_size = max(len(pps) // 10, 100)
            for i in tqdm(range(0, len(pps), chunk_size), desc=os.path.basename(file)):
                chunk = '\n\n'.join(pps[i:min(i + chunk_size, len(pps))])
                encoded = tokenizer.encode(chunk)
                num_tokens += len(encoded.tokens)

        return num_tokens

    token_count = 0
    byte_count = 0

    # Keep reading text files until we have num_bytes or run out of files.
    lang_files = [f for f in os.listdir(corpus_dir / lang) if 'truncated' not in f]
    random.shuffle(lang_files)
    for fname in lang_files:
        filesize = os.path.getsize(corpus_dir / lang / fname)
        if byte_count + filesize <= NUM_BYTES:
            token_count += count_tokens(str(corpus_dir / lang / fname))
            byte_count += filesize
        else:
            # truncate to get partial file of the right size
            wanted_filesize = NUM_BYTES - byte_count
            trunc_fname = f'{fname[:-4]}_truncated_{wanted_filesize}.txt'
            os.system(f'cp {corpus_dir / lang / fname} {corpus_dir / lang / trunc_fname}')
            trunc_filesize = truncate_file(corpus_dir / lang / trunc_fname, wanted_filesize)

            token_count += count_tokens(str(corpus_dir / lang / trunc_fname))
            byte_count += trunc_filesize
            # os.remove(str(corpus_dir / lang / trunc_fname))

        if byte_count >= NUM_BYTES:
            break

    with open(output_dir / 'token_byte_counts.json', 'w') as fout:
        d = {
            'token_count': token_count,
            'byte_count': byte_count
        }
        json.dump(d, fout, indent=5)

    print(f'Saved to {output_dir / "token_byte_counts.json"}', flush=True)


if __name__ == '__main__':
    main()
