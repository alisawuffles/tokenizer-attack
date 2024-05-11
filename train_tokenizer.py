import os
from pathlib import Path
import time
from utils import ensure_dir, train_tokenizer_or_dump_frequencies

corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus/processed')
trained_tokenizers_dir = Path('tokenizer_json/oscar_languages')

language_code = 'en'
num_files = 2
print(f'-- {language_code} --', flush=True)

# get text data
print('Loading text data...', flush=True)
text_files = []
for f in os.listdir(corpus_dir / f'{language_code}'):
    if f.endswith('.txt'):
        text_files.append(str(corpus_dir / f'{language_code}/{f}'))
    if num_files and len(text_files) >= num_files:
        break

print('Training tokenizer...', flush=True)
start_time = time.time()
tokenizer = train_tokenizer_or_dump_frequencies(text_files)
print(f'Train time: {time.time() - start_time}', flush=True)

ensure_dir(trained_tokenizers_dir / language_code)
tokenizer.model.save(str(trained_tokenizers_dir / language_code))
print('Tokenizer files saved to ' + str(trained_tokenizers_dir / language_code), flush=True)
