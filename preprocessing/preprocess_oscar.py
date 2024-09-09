"""
First, download the OSCAR corpus from https://huggingface.co/datasets/oscar-corpus/OSCAR-2301.
"""

import os
from pathlib import Path
import pandas as pd
from utils import ensure_dir


corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus')
clean_dir = corpus_dir / 'processed'
for d in os.listdir(corpus_dir):  # for each langauge directory
    if os.path.isdir(corpus_dir / d) and d.endswith('_meta'):
        language_code = d.split('_')[0]
        for f in os.listdir(corpus_dir / f'{language_code}_meta'):  # for each file
            if f.endswith('.zst'):
                file_to_process = corpus_dir / f'{language_code}_meta/{f}'
                print(f'Processing file {file_to_process}', flush=True)
                df = pd.read_json(file_to_process, lines=True)

                text = '\n\n'.join(df['content'])
                f_name = os.path.basename(file_to_process).split('.')[0]  # just the filename without extension
                language_code = f_name.split('_')[0]

                ensure_dir(clean_dir / language_code)
                with open(clean_dir / language_code / f'{f_name}.txt', 'w') as fo:
                    fo.write(text)

                print(f'Saved to {clean_dir}/{language_code}/{f_name}.txt', flush=True)

                del df
                del text
