import os
import json
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits, ByteLevel, Metaspace, Punctuation, Split
from tokenizers import normalizers
from tokenizers.normalizers import NFC


llm_normalizers = {
    'commandr': [NFC()]
}

llm_pretokenizers = {
    'llama': [
        Metaspace(prepend_scheme='first'),
        Punctuation(behavior='contiguous'),
        Digits(individual_digits=True)
    ],
    'llama3': [
        Split(
            pattern="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            behavior='isolated',
            invert=False
        ),
        Digits(),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    ],
    'gpt4': [
        Digits(),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    ],
    'gemma': [
        Metaspace(prepend_scheme='never'),
        Punctuation(behavior='contiguous'),
        Digits(individual_digits=True),
        Split(pattern='\n', behavior='removed'),
        Split(pattern='\t', behavior='removed')
    ],
    'commandr': [
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    ]
}


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_tokenizer_json(path_to_json):
    with open(path_to_json, 'r') as fin:
        tokenizer_json = json.load(fin)
        merges = tokenizer_json['model']['merges']

    return {
        'vocab': tokenizer_json['model']['vocab'],
        'merges': merges
    }


def read_merges_txt(path_to_txt):
    with open(path_to_txt) as fin:
        merges = fin.readlines()[1:]
        merges = [m.rsplit('\n', 1)[0] for m in merges]
    return {
        'merges': merges
    }


def train_tokenizer_or_dump_frequencies(text_files: str, model_name=None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], show_progress=True)

    if not model_name:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Digits(),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
        ])
    else:
        if model_name in llm_normalizers:
            tokenizer.normalizer = normalizers.Sequence(llm_normalizers[model_name])
        if model_name in llm_pretokenizers:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(llm_pretokenizers[model_name])

    tokenizer.train(text_files, trainer)

    return tokenizer
