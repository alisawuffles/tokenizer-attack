import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
import time
import random
from utils import ensure_dir

num_files = 100

while True:
    corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus/processed')
    trained_tokenizers_dir = Path('tokenizer_json/oscar_languages')

    # languages left
    languages_left = []
    for language_code in os.listdir(corpus_dir):
        if not os.path.exists(trained_tokenizers_dir / language_code):
            languages_left.append(language_code)

    language_code = random.sample(languages_left, 1)[0]
    print(f'-- {language_code} --', flush=True)

    # get text data
    print('Loading text data...', flush=True)
    text_files = []
    for f in os.listdir(corpus_dir / f'{language_code}'):
        if f.endswith('.txt'):
            text_files.append(str(corpus_dir / f'{language_code}/{f}'))
        if num_files and len(text_files) >= num_files:
            break

    print('Initializing tokenizer...', flush=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True)
    tokenizer.pre_tokenizer = Whitespace()

    print('Training tokenizer...', flush=True)
    start_time = time.time()
    tokenizer.train(text_files, trainer)
    print(f'Train time: {time.time() - start_time}', flush=True)

    ensure_dir(trained_tokenizers_dir / language_code)
    tokenizer.model.save(str(trained_tokenizers_dir / language_code))
    print('Saved!', flush=True)
