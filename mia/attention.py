import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from .utils import model, tokenizer, tokens_to_strs

with open('diffusion.json', 'r') as f:
    text = f.read()
    training_data = json.loads(text)

def get_tokens(s):
    input_ids = tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids
    tokens = tokens_to_strs(tokenizer.convert_ids_to_tokens(input_ids[0]))
    return tokens

def get_attentions(src, dest):
    src_ids = tokenizer(src, return_tensors="pt", add_special_tokens=True).input_ids
    dest_ids = tokenizer(dest, return_tensors="pt", add_special_tokens=True).input_ids
    outputs = model(input_ids=dest_ids, decoder_input_ids=src_ids)
    layer_attn = []
    for layer in outputs.cross_attentions:
        layer_attns = layer.squeeze(0)
        layer_attn.append(layer_attns)
    attns = torch.permute(torch.stack([x.detach() for x in layer_attn]), (2, 3, 0, 1))
    return attns[1:-1, 1:-1]

def generate_training():
    for case in tqdm(training_data[1000:]):
        src = case['src']
        src_tokens = get_tokens(src)
        dest = case['dest']
        dest_tokens = get_tokens(dest)
        attns = get_attentions(src, dest)
        matches = set(map(tuple, case['matches']))
        for src_idx in range(len(src_tokens)):
            for dest_idx in range(len(dest_tokens)):
                scores = attns[src_idx][dest_idx].numpy().flatten().tolist()
                src_token = src_tokens[src_idx]
                dest_token = dest_tokens[dest_idx]
                pair = (src_token, dest_token, scores, 1 if (src_idx, dest_idx) in matches else 0)
                yield pair

# for s, d, xs, y in generate_training():
#     print(','.join(map(str, ['"%s"' % s, '"%s"' % d] + xs + [y])))
