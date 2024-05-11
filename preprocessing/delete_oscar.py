import os
from pathlib import Path

corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus')
clean_dir = corpus_dir / 'processed'
for d in os.listdir(corpus_dir):
    if os.path.isdir(corpus_dir / d) and d.endswith('_meta'):
        language_code = d.split('_')[0]
        for f in os.listdir(corpus_dir / f'{language_code}_meta'):
            if f.endswith('.zst'):
                f_name = f.split('.')[0]
                if os.path.exists(clean_dir / f'{language_code}/{f_name}.txt'):
                    print(f'Removing {language_code}_meta/{f}')
                    os.remove(corpus_dir / f'{language_code}_meta/' / f)
