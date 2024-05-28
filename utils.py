import os
import json
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits, ByteLevel, Metaspace, Split
from tokenizers.normalizers import NFKC
from tokenizers import Regex
from constants import LLAMA_BASE_VOCAB
from typing import Dict, List, Tuple
from sentencepiece import SentencePieceProcessor
from tqdm import trange, tqdm


llm_normalizers = {
    'claude': NFKC()
}


llm_pretokenizers = {
    'bloom': pre_tokenizers.Sequence([
        Split(
            pattern=Regex(" ?[^(\\s|[.,!?…。，、।۔،])]+"),
            behavior='isolated', invert=False
        ),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    ]),
    'llama': pre_tokenizers.Sequence([
        Split(
            pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
            behavior='isolated', invert=False
        ),
        Metaspace(replacement='▁', prepend_scheme='first'),
        Digits(individual_digits=True),
        Split(pattern='\n', behavior='removed'),  # \n and \t never merged with anything
        Split(pattern='\t', behavior='removed'),
    ]),
    'llama3': pre_tokenizers.Sequence([
        Split(
            pattern=Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}T| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior='isolated',
            invert=False
        ),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    ]),
    'gpt2': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    'gpt3': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    'gpt3.5': pre_tokenizers.Sequence([
        Split(
            pattern=Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior='removed',
            invert=True
        ),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    ]),
    'gpt4o': pre_tokenizers.Sequence([
        Split(
            pattern=Regex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior='removed',
            invert=True
        ),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ]),
    'gemma': pre_tokenizers.Sequence([
        Split(
            pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
            behavior='isolated', invert=False
        ),
        Metaspace(prepend_scheme='never'),
        Digits(individual_digits=True),
        Split(pattern='\n', behavior='removed'),
        Split(pattern='\t', behavior='removed')
    ]),
    'commandr': pre_tokenizers.Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    ]),
    'mixtral': pre_tokenizers.Sequence([
        Split(
            pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
            behavior='isolated', invert=False
        ),
        Metaspace(prepend_scheme='first'),
        Digits(individual_digits=True),
        Split(pattern='\n', behavior='removed'),
        Split(pattern='\t', behavior='removed'),
    ]),
    'claude': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
}


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_tokenizer_json(path_to_json):
    with open(path_to_json, 'r') as fin:
        tokenizer_json = json.load(fin)

    return {
        'vocab': tokenizer_json['model']['vocab'],
        'merges': tokenizer_json['model']['merges']
    }


def read_merges_txt(path_to_txt):
    with open(path_to_txt) as fin:
        merges = fin.readlines()[1:]
        merges = [m.rsplit('\n', 1)[0] for m in merges]
    return merges


def get_size_cpe(text):
    """
    For tokenizers that are really *character*-pair encoding, we need to get the size in terms of the base vocabulary.
    If a character is in the base vocabulary, that adds 1 to the size. Else, it is the number of bytes.
    """
    size = 0
    for t in text:
        if t in LLAMA_BASE_VOCAB:
            size += 1
        else:
            size += len(t.encode('utf8'))
    return size


def train_tokenizer_or_dump_frequencies(text_files: str, model_name=None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], show_progress=True)

    if not model_name:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Digits(),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
        ])
    else:
        print(f'Using config of {model_name}')
        if model_name in llm_normalizers:
            tokenizer.normalizer = llm_normalizers[model_name]
        if model_name in llm_pretokenizers:
            tokenizer.pre_tokenizer = llm_pretokenizers[model_name]
        else:
            raise ValueError(f'Unknown model name: {model_name}')

    tokenizer.train(text_files, trainer)

    return tokenizer


def train_tokenizer_spm(text_files, output_dir):
    import sentencepiece as spm
    from utils import SentencePieceExtractor
    from os import linesep

    ensure_dir(output_dir / 'spm')
    spm.SentencePieceTrainer.train(
        input=text_files,
        model_prefix=str(output_dir / 'spm/m'),
        vocab_size=30000,
        character_coverage=0.9995,
        model_type='bpe',
        num_threads=32,
        train_extremely_large_corpus=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(output_dir / 'spm/m.model'))
    extractor = SentencePieceExtractor(model='data/test_spm/m.model')
    vocab, merges = extractor.extract()
    with open(output_dir / 'vocab.json', 'w') as vocab_f:
        json.dump(vocab, vocab_f)
    with open(output_dir / 'merges.txt', 'w') as merges_f:
        merges_f.write(f'# trained with SentencePiece{linesep}')
        merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{linesep}", merges))


def is_valid_unicode(data):
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def truncate_file(filename, wanted_filesize):
    if os.path.getsize(filename) < wanted_filesize:
        raise ValueError('File is already smaller than desired filesize')

    with open(filename, 'rb') as f:
        f.seek(wanted_filesize)
        data = f.read(1)
        while data and not is_valid_unicode(data):
            data = f.read(1)
            wanted_filesize += 1
    with open(filename, 'r+', encoding='utf-8') as fin:
        fin.truncate(wanted_filesize)

    # if we ever need to rerun all our experiments (god forbid), we should return the actual
    # filesized used and record that instead
    # return wanted_filesize


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. Taken from here:
    https://github.com/huggingface/tokenizers/blob/f2ec3b239b0a7a9866b01ec5cbd4d44243a40a16/bindings/python/scripts/sentencepiece_extractor.py#L17
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges
