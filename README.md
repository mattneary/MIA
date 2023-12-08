# MIA

> Mechanistic Interpretability of Attention

This project trains a logistic regression model on attention scores to make a
phrase matching predictor. Currently, the training data is limited. It's only a
set of paragraphs pairing the original order with a shuffling of sentences.
Token-to-token attention scores from BART (`num_heads * num_layers` per pair)
are then associated with a boolean that's true for corresponding tokens and
false for all others. The resulting model successfully generalizes from training
to test data.

Next steps are to enrich the training data with more sophisticated examples, mostly
rephrasing examples.

It also makes sense to expand the current test data with examples that were not
generated through sentence shuffling.

## Setup

```sh
$ poetry install
$ poetry run python -m spacy download en
$ poetry run python -m mia
```
