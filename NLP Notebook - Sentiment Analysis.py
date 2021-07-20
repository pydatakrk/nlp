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
zz = [len(x.text) for x in train_set]

import matplotlib.pyplot as plt

plt.hist(zz, bins=100, density=1)

 # %%
 # data.BucketIterator??

# %%
len(max(train_set, key=lambda x: len(x.text)).text)

# but we can do better!
bb = data.BucketIterator(train_set, batch_size=64, device=device, sort_key=lambda x: len(x.text), sort=True)
bb2 = [x for x in bb]

# %%
# data.BucketIterator??

# %%
# bb2[0].text.numpy.apply(text_field.vocab.itos)
import numpy

# ll = lambda x: text_field.vocab.itos[x]
# nn = numpy.vectorize(ll)
# for ii in range(64):
#     print(''.join(nn(bb2[0].text.numpy())[..., ii]))
#     print()

# %%
# # data.get_tokenizer??
# nlp = spacy.load("en_core_web_sm")
# gg = nlp("I am a dog in Paris! I am a cat.")
# gg.to_json()
