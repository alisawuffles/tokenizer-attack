"""
Normalizer and pretokenizer configurations for LLM tokenizers.
"""

from tokenizers import pre_tokenizers, Regex
from tokenizers.pre_tokenizers import Digits, ByteLevel, Metaspace, Split
from tokenizers.normalizers import NFKC, NFC


LLM_NORMALIZERS = {
    'claude': NFKC(),
    'command-r': NFC(),
    'gpt-neox': NFC(),
}


LLM_PRETOKENIZERS = {
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
    'gpt-neox': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    'gpt2': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
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
    'command-r': pre_tokenizers.Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    ]),
    'mistral': pre_tokenizers.Sequence([
        Split(
            pattern=Regex(" ?[^\\s\\p{L}\\p{N}]+\\r*"),
            behavior='isolated',
            invert=False
        ),
        Metaspace(prepend_scheme='first'),
        Digits(individual_digits=True),
        Split(pattern='\n', behavior='removed'),
        Split(pattern='\t', behavior='removed'),
    ]),
    'mistral-nemo': pre_tokenizers.Sequence([
        Split(
            pattern=Regex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior='isolated',
            invert=False,
        ),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ]),
    'claude': ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
}
