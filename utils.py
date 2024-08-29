from __future__ import annotations

import heapq
import itertools
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import simdjson as json
import tqdm.auto as tqdm
from sentencepiece import SentencePieceProcessor
from tokenizers import Regex, Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel, Digits, Metaspace, Split
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm, trange

from constants import LLAMA_BASE_VOCAB

llm_normalizers = {"claude": NFKC()}


llm_pretokenizers = {
    "bloom": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(" ?[^(\\s|[.,!?…。，、।۔،])]+"),
                behavior="isolated",
                invert=False,
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    ),
    "llama": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
                behavior="isolated",
                invert=False,
            ),
            Metaspace(replacement="▁", prepend_scheme="first"),
            Digits(individual_digits=True),
            Split(
                pattern="\n", behavior="removed"
            ),  # \n and \t never merged with anything
            Split(pattern="\t", behavior="removed"),
        ]
    ),
    "llama3": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}T| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                behavior="isolated",
                invert=False,
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    ),
    "gpt2": ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    "gpt3": ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    "gpt3.5": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                behavior="removed",
                invert=True,
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    ),
    "gpt4o": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                behavior="removed",
                invert=True,
            ),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    ),
    "gemma": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
                behavior="isolated",
                invert=False,
            ),
            Metaspace(prepend_scheme="never"),
            Digits(individual_digits=True),
            Split(pattern="\n", behavior="removed"),
            Split(pattern="\t", behavior="removed"),
        ]
    ),
    "commandr": pre_tokenizers.Sequence(
        [
            Digits(individual_digits=True),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
        ]
    ),
    "mixtral": pre_tokenizers.Sequence(
        [
            Split(
                pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
                behavior="isolated",
                invert=False,
            ),
            Metaspace(prepend_scheme="first"),
            Digits(individual_digits=True),
            Split(pattern="\n", behavior="removed"),
            Split(pattern="\t", behavior="removed"),
        ]
    ),
    "claude": ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
}


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_tokenizer_json(path_to_json):
    with open(path_to_json, "r") as fin:
        tokenizer_json = json.load(fin)

    return {
        "vocab": tokenizer_json["model"]["vocab"],
        "merges": tokenizer_json["model"]["merges"],
    }


def read_merges_txt(path_to_txt):
    with open(path_to_txt) as fin:
        merges = fin.readlines()[1:]
        merges = [m.rsplit("\n", 1)[0] for m in merges]
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
            size += len(t.encode("utf8"))
    return size


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
        print(f"Using config of {model_name}")
        if model_name in llm_normalizers:
            tokenizer.normalizer = llm_normalizers[model_name]
        if model_name in llm_pretokenizers:
            tokenizer.pre_tokenizer = llm_pretokenizers[model_name]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    tokenizer.train(text_files, trainer)

    return tokenizer


def train_tokenizer_spm(text_files, output_dir):
    from os import linesep

    import sentencepiece as spm

    from utils import SentencePieceExtractor

    ensure_dir(output_dir / "spm")
    spm.SentencePieceTrainer.train(
        input=text_files,
        model_prefix=str(output_dir / "spm/m"),
        vocab_size=30000,
        character_coverage=0.9995,
        model_type="bpe",
        num_threads=32,
        train_extremely_large_corpus=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(output_dir / "spm/m.model"))
    extractor = SentencePieceExtractor(model="data/test_spm/m.model")
    vocab, merges = extractor.extract()
    with open(output_dir / "vocab.json", "w") as vocab_f:
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
    if os.path.getsize(filename) < wanted_filesize:
        raise ValueError("File is already smaller than desired filesize")

    with open(filename, "rb") as f:
        f.seek(wanted_filesize)
        data = f.read(1)
        while data and not is_valid_unicode(data):
            data = f.read(1)
            wanted_filesize += 1
    with open(filename, "r+", encoding="utf-8") as fin:
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


CODE_LANGS = [
    "Fortran",
    "Perl",
    "Motorola68KAssembly",
    "Ruby",
    "XML",
    "reStructuredText",
    "PowerShell",
    "Batchfile",
    "Smali",
    "VisualBasic.NET",
    "Pod6",
    "Makefile",
    "Lua",
    "JavaScript",
    "Hack",
    "Scala",
    "HTML",
    "XPixMap",
    "Python",
    "PHP",
    "CMake",
    "TSQL",
    "Haskell",
    "C++",
    "C",
    "CSS",
    "Dockerfile",
    "Objective-C",
    "Raku",
    "Java",
    "Smalltalk",
    "FortranFreeForm",
    "Shell",
    "TeX",
    "Julia",
    "Markdown",
    "Go",
]
DOMAIN_LANGS = ["arxiv", "books", "github", "web", "wikipedia"]
OTHER_LANGS = [
    "fy",
    "sah",
    "ga",
    "sa",
    "os",
    "cv",
    "ceb",
    "af",
    "br",
    "azb",
    "hr",
    "mhr",
    "lb",
    "uz",
    "ce",
    "mg",
    "nds",
    "xmf",
    "bpy",
    "new",
    "min",
    "arz",
    "nn",
    "tk",
    "pms",
    "ms",
    "gom",
    "la",
    "jbo",
    "mt",
    "sw",
]
DO_COUNT_CHARS = {
    "gpt2": False,
    "gpt3": False,
    "gpt3.5": False,
    "gpt4o": False,
    "llama": True,
    "llama3": False,
    "mixtral": True,
    "gemma": True,
}
LLM_LANGS = [
    "fa",
    "as",
    "no",
    "hi",
    "ur",
    "mr",
    "tr",
    "jbo",
    "ar",
    "ps",
    "mn",
    "pnb",
    "arz",
    "lv",
    "tt",
    "ne",
    "sq",
    "sw",
    "bpy",
    "min",
    "sk",
    "dv",
    "ms",
    "fy",
    "azb",
    "gom",
    "sa",
    "new",
    " sd",
    "ka",
    "or",
    "pa",
    "bn",
    "my",
    "ta",
    "bo",
    "he",
    "ml",
    "yi",
    "si",
    "te",
    "ckb",
    "gu",
    "kn",
    "ug",
    "ceb",
    "ky",
    "xmf",
    "hy",
    "tk",
    "el",
    "mhr",
    "lt",
    "km",
    "lo",
    "sah",
    "zh",
    "os",
    "ku",
    "la",
    "ba",
    "th",
    "hr",
    "kk",
    "eu",
    "mt",
    "az",
    "tg",
    "am",
    "uz",
    "is",
    "ce",
    "vi",
    "et",
    "cv",
    "tl",
    "mg",
    "id",
    "pms",
    "mk",
    "br",
    "lb",
    "cy",
    "hu",
    "ko",
    "be",
    "ga ",
    "af",
    "sl",
    "fi",
    "bg",
    "eo",
    "nn",
    "da",
    "gl",
    "ja",
    "ro",
    "sr",
    "nds",
    "cs",
    "sv",
    "ca",
    "pt",
    "nl",
    "uk",
    "pl",
    "it",
    "ru",
    "es",
    "de",
    "fr",
    "wikipedia",
    "arxiv",
    "web",
    "github",
    "books",
]


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
            # item.name in CODE_LANGS
            # # or item.name in OTHER_LANGS
            # or item.name in ["code", "en", "wikipedia_uncleaned", "c4", "other"]
        ):
            continue

        lang = item.name
        if subdir is not None:
            item = item / subdir

        # if (
        #     item.name in CODE_LANGS
        #     or item.name in OTHER_LANGS
        #     or (item.name in DOMAIN_LANGS and item.name != "github")
        #     or item.name == "code"
        # ):
        #     continue
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
