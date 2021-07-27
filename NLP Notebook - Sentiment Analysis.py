# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# # !python -m spacy download en_core_web_sm
# import spacy
# spacy.load('en_core_web_sm')

# %%
import spacy
import torch
import torchtext

from torchtext.legacy import datasets, data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Containers for tokenisation
# using tokenize="spacy" because it's the best.
text_field = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm")
label_field = data.LabelField(dtype=torch.float) # torch.float because GPUs use floats

# Load dataset and split to train and test data
# IMDB dataset (about movies)
train, test = datasets.IMDB.splits(text_field=text_field, label_field=label_field)

# %%
# Split to train and validation set - 80% to train_set, 20% to validation_set
# The original set is 25k descriptions(?) so train_set after the split is 20k and valid_set is 5k.
train_set, valid_set = train.split(0.8)
len(train_set), len(valid_set)  # 20_000, 5_000
text_field.build_vocab(train_set, max_size=25_000)
label_field.build_vocab(train_set)

assert len(text_field.vocab) == 25_002

# %%
# Map int to string and string to int
# text_field.vocab.itos[186] -> 'though'
# text_field.vocab.stoi['though'] -> 186

# %%
text_field.vocab.itos[:10]

# %%
zz = [len(x.text) for x in train_set]

import matplotlib.pyplot as plt

plt.hist(zz, bins=100, density=1)

# %%
len(max(train_set, key=lambda x: len(x.text)).text)

# but we can do better!
train_buckets, valid_buckets, test_buckets = data.BucketIterator.splits(
    (train_set, valid_set, test), batch_size=64, device=device
)

# %%
from torch import nn


class NLPModule(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_size, out_features):
        # before parent
        super().__init__()
        # after parent
        # warstwa osadzeń/osadzanie(?) embedding
        # wektory w przestrzeni znaczeniowej słów
        self.embedding = nn.Embedding(num_embedding, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, input):
        embed_output = self.embedding(input)
        rnn_output, hidden_output = self.rnn(embed_output)
        # hidden_output is the same as rnn_output[-1]
        lin_output = self.linear(hidden_output)

        return lin_output


# %%
>>> rnn = nn.RNN(3, 2, 1)
>>> input = torch.randn(5, 3, 3)
>>> h0 = torch.randn(1, 3, 2)
>>> output, hn = rnn(input, h0)
output, hn

# %%
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(100, 19)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,99,1,0, 4,3,2,9],[4,3,2,9, 4,3,2,9]])
>>> embedding(input)
