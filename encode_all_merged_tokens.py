from utils import read_tokenizer_json
from pathlib import Path
from tokenizers import Tokenizer
import os
from tqdm import tqdm


if os.path.exists('used_merges/llama3_ordered_used_merges.txt'):
    os.remove('used_merges/llama3_ordered_used_merges.txt')
Path('used_merges/llama3_ordered_used_merges.txt').touch()
tokenizer = Tokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer_dir = Path('tokenizer_json/llm_tokenizers')
tokenizer_json = read_tokenizer_json(tokenizer_dir / 'llama3_tokenizer.json')
token_mapping = {}
for k, v in tokenizer_json['vocab'].items():
    if len(k) == 1:
        new_k = tokenizer.decode([tokenizer.token_to_id(k)])
        if new_k != k:
            token_mapping[k] = new_k

token_creation_order = []
for m in tqdm(tokenizer_json['merges']):
    merged_token = ''.join(m.split(' '))
    for k, v in token_mapping.items():
        merged_token = merged_token.replace(k, v)
    if any([c == '�' for c in merged_token]):
        continue
    if len(token_creation_order) == 0 or token_creation_order[-1] != merged_token:
        token_creation_order.append(merged_token)

for word in token_creation_order:
    tokenizer.encode_batch([word])
