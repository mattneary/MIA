import spacy
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
nlp = spacy.load("en_core_web_sm")

def tokens_to_strs(tokens):
    running_str = ''
    strs = []
    for idx, token in enumerate(tokens):
        joined_str = tokenizer.convert_tokens_to_string(tokens[:(idx+1)])
        incremental_str = joined_str[len(running_str):]
        strs.append(incremental_str)
        running_str = joined_str
    return strs
