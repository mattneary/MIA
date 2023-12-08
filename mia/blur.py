import numpy as np
import json
from tqdm import tqdm
from .utils import nlp, tokenizer, tokens_to_strs

with open('gatsby.txt', 'r') as f:
    text = f.read()
    blocks = [x.strip() for x in text.split('\n\n') if x.strip()]

def get_sentences(block):
    doc = nlp(block)
    sentences = list(doc.sents)
    ranges = []
    for sent in sentences:
        ranges.append((sent.start_char, sent.end_char))
    return ranges

def shuffle_block(block, ordering=None):
    doc = nlp(block)
    sentences = list(doc.sents)
    ordering = ordering or np.random.permutation(len(sentences)).tolist()

    shuffled = ''
    links = []
    for j, i in enumerate(ordering):
        sent = sentences[i]
        last_sent = j == len(sentences) - 1
        insertion = str(doc[sent.start:sent.end])
        # need to look back to account for the leading space on tokens
        src_range = ((-1 if j else 0) + len(shuffled), len(shuffled) + len(insertion))
        dest_range = ((-1 if i else 0) + sent.start_char, sent.end_char)
        links.append((src_range, dest_range))
        shuffled += insertion + ('' if last_sent else ' ')

    return shuffled, links

def tokens_in_range(token_strs, start, end):
    running_len = 0
    for token_idx, token_str in enumerate(token_strs):
        rng = (running_len, running_len + len(token_str))
        if rng[0] >= start and rng[1] <= end:
            yield token_idx, token_str
        running_len += len(token_str)

def block_to_tokens(block):
    input_ids = tokenizer(block, return_tensors="pt", add_special_tokens=False).input_ids
    tokens = tokens_to_strs(tokenizer.convert_ids_to_tokens(input_ids[0]))
    return tokens

def generate_training():
    training_data = []
    for block in tqdm(blocks):
        for _ in range(0, 3):
            output, links = shuffle_block(block)
            block_tokens = block_to_tokens(block)
            output_tokens = block_to_tokens(output)
            matches = []
            for (src_start, src_end), (dest_start, dest_end) in links:
                dest_tokens = list(tokens_in_range(block_tokens, dest_start, dest_end))
                src_tokens = list(tokens_in_range(output_tokens, src_start, src_end))
                token_pairs = list(reversed([
                    (src_idx, dest_idx)
                    for (src_idx, src_token), (dest_idx, dest_token)
                    in zip(reversed(src_tokens), reversed(dest_tokens))
                    if src_token == dest_token
                ]))
                matches += token_pairs
            training_data.append({
                'src': output,
                'dest': block,
                'matches': matches,
            })
    return training_data

# print(json.dumps(generate_training()))
