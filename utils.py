from __future__ import annotations

import heapq
import itertools
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import simdjson as json
import tqdm.auto as tqdm
from sentencepiece import SentencePieceProcessor
from tokenizers.models import BPE

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.pre_tokenizers import Digits, ByteLevel
from tokenizers.trainers import BpeTrainer
from constants import LLM_LANGS
from llm_tokenizer_configs import LLM_NORMALIZERS, LLM_PRETOKENIZERS



def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_json(file):
    return json.load(open(file))


def read_predictions(solution_file):
    return read_json(solution_file)['lang_vals']


def read_tokenizer_json(path_to_json):
    tokenizer_json = read_json(path_to_json)

    return {
        "vocab": tokenizer_json["model"]["vocab"],
        "merges": tokenizer_json["model"]["merges"],
    }


def read_merges_txt(path_to_txt):
    with open(path_to_txt) as fin:
        merges = fin.readlines()[1:]
        merges = [m.rsplit("\n", 1)[0] for m in merges]
    return merges


def train_tokenizer_or_dump_frequencies(text_files: str, model_name=None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], show_progress=True)

    if not model_name:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                Digits(),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
            ]
        )
    else:
        print(f'Using config of {model_name}')
        if model_name in LLM_NORMALIZERS:
            tokenizer.normalizer = LLM_NORMALIZERS[model_name]
        if model_name in LLM_PRETOKENIZERS:
            tokenizer.pre_tokenizer = LLM_PRETOKENIZERS[model_name]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    tokenizer.train(text_files, trainer)

    return tokenizer


def train_tokenizer_spm(text_files, output_dir):
    import sentencepiece as spm
    from utils import SentencePieceExtractor
    from os import linesep
    from clean_unused_merges import clean_merges

    spm.SentencePieceTrainer.train(
        input=text_files,
        model_prefix=str(output_dir / 'spm'),
        vocab_size=30000,
        character_coverage=0.995,
        model_type='bpe',
        num_threads=32,
        train_extremely_large_corpus=True,
        byte_fallback=True,
        split_digits=True,
        add_dummy_prefix=True
    )

    extractor = SentencePieceExtractor(model=str(output_dir / 'spm.model'))
    vocab, merges = extractor.extract()
    merges = clean_merges(merges)

    with open(output_dir / 'vocab.json', 'w') as vocab_f:
        json.dump(vocab, vocab_f)
    with open(output_dir / "merges.txt", "w") as merges_f:
        merges_f.write(f"# trained with SentencePiece{linesep}")
        merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{linesep}", merges))


def is_valid_unicode(data):
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def truncate_file(filename, wanted_filesize):
    """
    This truncates filename to wanted_filesize. Note this overwrites filename!!
    """
    if os.path.getsize(filename) < wanted_filesize:
        raise ValueError("File is already smaller than desired filesize")

    # adjust wanted_filesize to the next valid unicode character
    with open(filename, "rb") as f:
        f.seek(wanted_filesize)
        data = f.read(1)
        while data and not is_valid_unicode(data):
            data = f.read(1)
            wanted_filesize += 1

    with open(filename, "r+", encoding="utf-8") as fin:
        fin.truncate(wanted_filesize)

    return wanted_filesize


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
        vocab = {sp.id_to_piece(index): index for index in tqdm.trange(sp.GetPieceSize())}

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


def bytes_to_unicode():
    """
    MJ: STOLEN DIRECTLY FROM https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    --------------
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class PriorityQueue:
    def __init__(self, items=None, max_queue=True):
        self.pq = []
        self.removed = object()
        self.entry_finder = {}
        self.counter = itertools.count()
        self.max_queue = max_queue
        if items is not None:
            for el, priority in items:
                if self.max_queue:
                    priority = -priority
                assert el not in self
                count = next(self.counter)
                entry = [priority, count, el]
                self.entry_finder[el] = entry
                self.pq.append(entry)
            heapq.heapify(self.pq)

    def add(self, el, priority):
        if self.max_queue:
            priority = -priority
        if el in self:
            self.remove(el)
        count = next(self.counter)
        entry = [priority, count, el]
        self.entry_finder[el] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, el):
        entry = self.entry_finder.pop(el)
        entry[-1] = self.removed

    def pop(self):
        while self.pq:
            priority, count, el = heapq.heappop(self.pq)
            if el is not self.removed:
                del self.entry_finder[el]
                if self.max_queue:
                    priority = -priority
                return el, priority
        raise KeyError("pop from an empty priority queue")

    def peek(self):
        while self.pq:
            priority, count, el = self.pq[0]
            if el is self.removed:
                heapq.heappop(self.pq)
                continue

            if self.max_queue:
                priority = -priority
            return el, priority
        raise KeyError("peek from an empty priority queue")

    def lookup(self, el, default=None):
        priority = self.entry_finder.get(el, (default,))[0]
        if self.max_queue:
            priority = -priority
        return priority

    def __getitem__(self, el):
        return self.entry_finder[el][0]

    def __contains__(self, el):
        return el in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)


@dataclass
class Merge:
    rank: int
    l: str
    r: str
    lp: Optional[Merge] = None
    rp: Optional[Merge] = None
    lc: list[Merge] = field(default_factory=list)
    rc: list[Merge] = field(default_factory=list)

    @property
    def m(self):
        return self.l + self.r

    @property
    def c(self):
        return self.lc + self.rc

    def __str__(self):
        return f"{self.l} {self.r}"

    def __repr__(self):
        return f"{self.l}≀{self.r}"


def postprocess_merges(merges):
    producers = {}
    merge_order = []
    for i, merge_str in enumerate(merges):
        try:
            left, right = merge_str.split(" ")
        except ValueError:
            print(f"Broken merge {i}: {merge_str!r}")
            break
        merge = Merge(i + 1, left, right)
        merge_order.append(merge)
        producers[merge.m] = merge
        merge.lp = producers.setdefault(merge.l, Merge(0, merge.l, ""))
        merge.rp = producers.setdefault(merge.r, Merge(0, merge.r, ""))
        if merge.lp is not None:
            merge.lp.lc.append(merge)
        if merge.rp is not None:
            merge.rp.rc.append(merge)

    return merge_order, producers


def load_merges(fname):
    with Path(fname).open() as f:
        return [line.rstrip("\n") for line in f.readlines()[1:]]


def load_langlist(root, name):
    with (root / f"{name}.txt").open() as f:
        return [root / lang.strip() for lang in f.read().strip().split()]


def load_data(data_root, verbose=False, subdir=None, langlist=None):
    merges, producer = postprocess_merges(load_merges(data_root / "merges.txt"))

    pair_counts = {}
    training_counts = {}
    P = partial(tqdm.tqdm, dynamic_ncols=True) if verbose else lambda x: x
    langlist = (
        list(data_root.iterdir())
        if langlist is None
        else load_langlist(data_root, langlist)
    )
    for item in P(langlist):
        if not item.is_dir() or item.name.startswith("."):
            continue

        if data_root.parent.name.startswith("llm") and (
            item.name
            not in LLM_LANGS
        ):
            continue

        lang = item.name
        if subdir is not None:
            item = item / subdir

        with (item / "all_pair_counts.json").open() as f:
            pair_counts[lang] = json.load(f)

        with (item / "meta.json").open() as f:
            data = json.load(f)
            for key in ["byte_count", "char_count", "count1", "count2"]:
                if key in data:
                    counter = training_counts.setdefault(key, {})
                    counter[lang] = data[key]

            counter = training_counts.setdefault("pairs", {})
            counter[lang] = sum(pair_counts[lang][0].values())

    print(training_counts.keys())
    return merges, pair_counts, training_counts


def get_pair_to_byte_ratios(ex_dir, num_bytes=None):
    pair_to_byte_ratio = {}
    meta = read_json(ex_dir / 'meta.json')

    for lang in meta['byte_count'].keys():
        # if the byte_count is different from default, the directory structure is a little different
        if num_bytes:
            lang_dir = Path(lang) / "{:.0e}".format(num_bytes).replace("e+", "e")
        else:
            lang_dir = lang

        lang_meta = read_json(ex_dir / lang_dir / 'meta.json')

        if 'pairs' in lang_meta:
            num_pairs = lang_meta['pairs']
        else:
            num_pairs = sum(read_json(ex_dir / lang_dir / 'all_pair_counts.json')[0].values())
            lang_meta['pairs'] = num_pairs

            # replace meta.json to include pair count
            with open(ex_dir / lang_dir / 'meta.json', 'w') as fo:
                json.dump(lang_meta, fo, indent=5)

        pair_to_byte_ratio[lang] = lang_meta['byte_count'] / num_pairs

    return pair_to_byte_ratio


def score_solution(test_dir, solution_file, num_bytes=None):
    """
    Return MSE for the given solution file, converting the predictions from pair counts to byte counts.
    """
    preds = read_predictions(test_dir / solution_file)
    meta = read_json(test_dir / 'meta.json')
    truth = {k: v / sum(meta['byte_count'].values()) for k, v in meta['byte_count'].items()}
    pair_to_byte_ratio = get_pair_to_byte_ratios(test_dir, num_bytes=num_bytes)

    converted_preds = {}
    for lang, p in preds.items():
        converted_preds[lang] = p * pair_to_byte_ratio[lang]
    converted_preds = {k: v / sum(converted_preds.values()) for k, v in converted_preds.items()}

    return mse(truth, converted_preds)


def mse(true, pred):
    return sum((pred[lang] - true[lang]) ** 2 for lang in true.keys()) / len(true)

