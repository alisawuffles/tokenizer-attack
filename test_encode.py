# import os
# from pathlib import Path
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# corpus_dir = Path('/gscratch/scrubbed/alisaliu/oscar-corpus/processed')
# language_code = 'en'
# for f in os.listdir(corpus_dir / f'{language_code}'):
#     if f.endswith('.txt'):
#         with open(corpus_dir / f'{language_code}/{f}', 'r') as fin:
#             texts = fin.readlines()
#     break

texts = ['âĢĻs']
tokenizer.encode_batch(texts)
