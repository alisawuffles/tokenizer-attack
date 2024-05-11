import os
from pathlib import Path
import random
import pandas as pd
from utils import ensure_dir


while True:
    corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus')
    clean_dir = corpus_dir / 'processed'
    unprocessed_files = []
    for d in os.listdir(corpus_dir):
        if os.path.isdir(corpus_dir / d) and d.endswith('_meta'):
            language_code = d.split('_')[0]
            for f in os.listdir(corpus_dir / f'{language_code}_meta'):
                if f.endswith('.zst'):
                    f_name = f.split('.')[0]
                    if not os.path.exists(clean_dir / f'{language_code}/{f_name}.txt'):
                        unprocessed_files.append(corpus_dir / f'{language_code}_meta/{f}')

    file_to_process = str(random.sample(unprocessed_files, 1)[0])

    print(f'Processing file {file_to_process}', flush=True)

    df = pd.read_json(file_to_process, lines=True)
    text = '\n\n'.join(df['content'])
    f_name = os.path.basename(file_to_process).split('.')[0]  # just the filename without extension
    language_code = f_name.split('_')[0]

    ensure_dir(clean_dir / language_code)
    with open(clean_dir / f'{language_code}/{f_name}.txt', 'w') as fo:
        fo.write(text)

    print(f'Saved to {clean_dir}/{language_code}/{f_name}.txt', flush=True)

    del df
    del text
