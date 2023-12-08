import json
from transformers import AutoTokenizer
from tqdm import tqdm
from .utils import tokenizer, tokens_to_strs

with open('diffusion.json', 'r') as f:
    text = f.read()
    training_data = json.loads(text)

def get_tokens(s):
    input_ids = tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids
    tokens = tokens_to_strs(tokenizer.convert_ids_to_tokens(input_ids[0]))
    return tokens

def validate():
    for case in tqdm(training_data):
        src = case['src']
        dest = case['dest']
        src_tokens = get_tokens(src)
        dest_tokens = get_tokens(dest)
        for src_idx, dest_idx in case['matches']:
            if src_tokens[src_idx] != dest_tokens[dest_idx]:
                print('mismatch')

#validate()
