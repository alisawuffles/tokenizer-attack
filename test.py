import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Metaspace, Punctuation
from utils import ensure_dir
import numpy as np
import json


def sample_from_simplex(n):
    """Return uniformly random vector in the n-simplex"""

    k = np.random.exponential(scale=1.0, size=n)
    return k / sum(k)


trained_tokenizers_dir = Path('/gscratch/xlab/alisaliu/hack-tokenizers/data/test')
ensure_dir(trained_tokenizers_dir)

weights = [1, 2]
alisa_file = str(trained_tokenizers_dir / 'I-am-Alisa.txt')
jon_file = str(trained_tokenizers_dir / 'I-am-Jon.txt')
text_files = [alisa_file] * weights[0] + [jon_file] * weights[1]

print('Initializing tokenizer...', flush=True)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]"], show_progress=True)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Metaspace(), Punctuation(behavior='contiguous')])

print('Training tokenizer...', flush=True)
tokenizer.train(text_files, trainer)

# filename contains both languages and corresponding weights
dirname = '0'
ensure_dir(trained_tokenizers_dir / dirname)
tokenizer.model.save(str(trained_tokenizers_dir / dirname))

with open(trained_tokenizers_dir / '0' / 'config.json', 'w') as fo:
    config = {}
    config['train_files'] = text_files
    json.dump(config, fo, indent=5)

print('Tokenizer info saved to ' + str(trained_tokenizers_dir / dirname), flush=True)


for lang_code in ['Alisa', 'Jon']:
    print('TRAINING TOKENIZER FOR ', lang_code, flush=True)
    os.chdir(trained_tokenizers_dir / dirname)
    print('Current directory:', os.getcwd(), flush=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], show_progress=False)
    tokenizer.pre_tokenizer = Whitespace()
    text_files = [alisa_file] * weights[0] if lang_code == 'Alisa' else [jon_file] * weights[1]
    tokenizer.train(text_files, trainer)
    ensure_dir(lang_code)
    tokenizer.model.save(lang_code)
    os.remove(f'{lang_code}/merges.txt')
    os.remove(f'{lang_code}/vocab.json')
