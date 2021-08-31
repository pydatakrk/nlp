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

# %% id="2bcf3010"
# # !python -m spacy download en_core_web_sm
# import spacy
# spacy.load('en_core_web_sm')

# %% id="9eab81be"
import spacy
import torch
import torchtext
from torchtext.legacy import datasets, data
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% id="f549101f"
# Containers for tokenisation
# using tokenize="spacy" because it's the best.
text_field = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", fix_length=100, batch_first=True)
label_field = data.LabelField(dtype=torch.float) # torch.float because GPUs use floats

# Load dataset and split to train and test data
# IMDB dataset (about movies)
train, test = datasets.IMDB.splits(text_field=text_field, label_field=label_field)

# %% id="8dc7770b"
# Split to train and validation set - 80% to train_set, 20% to validation_set
# The original set is 25k descriptions(?) so train_set after the split is 20k and valid_set is 5k.
train_set, valid_set = train.split(0.8)
len(train_set), len(valid_set)  # 20_000, 5_000
text_field.build_vocab(train_set, max_size=25_000, vectors="glove.6B.100d")
label_field.build_vocab(train_set)

assert len(text_field.vocab) == 25_002

# %% id="2d2abd5f"
# Map int to string and string to int
# text_field.vocab.itos[186] -> 'though'
# text_field.vocab.stoi['though'] -> 186

# %% id="6cc86523" colab={"base_uri": "https://localhost:8080/"} outputId="bfbd22be-78e2-4aea-a0a7-a9c9480c537a"
text_field.vocab.itos[:10]

# %% id="089e9316"
len(max(train_set, key=lambda x: len(x.text)).text)

# but we can do better!
train_buckets, valid_buckets, test_buckets = data.BucketIterator.splits(
    (train_set, valid_set, test), batch_size=64, device=device
)

# %% id="35a768f0"
from torch import nn
from typing import Tuple


class NLPModuleCNN(nn.Module):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        out_channels: int,
        kernel_size: Tuple[int],
        out_features: int,
        p_dropout: float,
        pad_index: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embedding, embedding_dim, padding_idx=pad_index
        )
        self.convo2d_0 = nn.Conv2d(1, out_channels, (kernel_size[0], embedding_dim))
        self.convo2d_1 = nn.Conv2d(1, out_channels, (kernel_size[1], embedding_dim))
        self.convo2d_2 = nn.Conv2d(1, out_channels, (kernel_size[2], embedding_dim))
        self.linear = nn.Linear(in_features=out_channels * 3, out_features=out_features)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, input):
        embed = self.embedding(input)
        c0 = F.relu(self.convo2d_0(embed))
        c1 = F.relu(self.convo2d_1(embed))
        c2 = F.relu(self.convo2d_2(embed))
        c0 = F.max_pool1d(c0)
        c1 = F.max_pool1d(c1)
        c2 = F.max_pool1d(c2)
        lin = self.linear(torch.cat([c0, c1, c2]))
        return lin


num_embedding = len(text_field.vocab)
embedding_dim = 100
hidden_size = 256
out_features = 1

model = NLPModuleCNN(
    num_embedding=num_embedding,
    embedding_dim=embedding_dim,
    out_channels=100,
    kernel_size=(2, 3, 7),
    out_features=out_features,
    p_dropout=0.5,
    pad_index=text_field.vocab.stoi[text_field.pad_token],
)

# %% colab={"base_uri": "https://localhost:8080/"} id="S5Ke57xrsby4" outputId="e8c521ca-399c-4533-f67d-6132e977106b"
pre_trained_embeddings = text_field.vocab.vectors
model.embedding.weight.data.copy_(pre_trained_embeddings)


# %% id="a3c6bc10" colab={"base_uri": "https://localhost:8080/"} outputId="bdb3ca18-2887-4fd5-8b0e-deebbed81fe6"
def policz(mod):
    return sum(p.numel() for p in mod.parameters())


policz(model)

# %% id="3fe5b41c"
# Stochastic gradient descent SGD
# minimalizować funkcję kosztu (szukanie minimum)

import torch.optim as optim

optimiser = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

# %% id="8058738f" colab={"base_uri": "https://localhost:8080/"} outputId="c9b718f2-9e52-475d-ab8a-427ddbc65de8"
ciretrion = criterion.to(device)
model = model.to(device)

def binary_accuracy(prediction, target):
    prediction = F.sigmoid(prediction)
    prediction = torch.round(prediction)
    
    compared = (prediction == target).float()
    return torch.mean(compared)


T = torch.tensor
binary_accuracy(T([0, 0.5, .2, 0.001, 0.8]), T([0, 1, 1, 1, 1]))

# %% id="21dc8651"
import numpy as np
import tqdm

def train(mod, data, optimiser, criterion):
    losses = []
    metrics = []
    mod = mod.to(device)

    # train pozwala na akumulację błędów, które potem będziemy propagować wstecz
    mod.train()

    for bucket in tqdm.tqdm(data):
        optimiser.zero_grad()
        output = mod(bucket.text).squeeze(0).squeeze(1)
        loss = criterion(output, bucket.label)
        metric = binary_accuracy(output, bucket.label)
        losses.append(loss.item())
        metrics.append(metric.item())
        loss.backward()
        optimiser.step()
        
        # print(np.mean(losses), losses[-1], np.mean(metrics), metrics[-1])

    return losses, metrics


# %% id="-vHoNh6cYlcW"
def validate(mod, data, criterion):
    losses = []
    metrics = []

    # wyłącza akumulacje błędów (z którego korzystaliśmy w train)
    mod.eval()

    for bucket in tqdm.tqdm(data):
        output = mod(bucket.text).squeeze(0).squeeze(1)
        loss = criterion(output, bucket.label)
        metric = binary_accuracy(output, bucket.label)
        losses.append(loss.item())
        metrics.append(metric.item())        
        # print(np.mean(losses), losses[-1], np.mean(metrics), metrics[-1])

    return losses, metrics



# %% id="SDhHPvnNbKnz" colab={"base_uri": "https://localhost:8080/", "height": 400} outputId="f40c362e-450e-4a12-aa2b-b6646adfa4bd"
for i in range(5):
  train_losses, train_metrics = train(model, train_buckets, optimiser, criterion)
  validated_losses, validated_metrics = validate(model, valid_buckets, criterion)
  
  print()
  print("Train metrics", np.mean(train_losses), np.mean(train_metrics))
  print("Validation metrics", np.mean(validated_losses), np.mean(validated_metrics))


# %% id="hvMObYJGhMCG"
tokenizer = spacy.load("en_core_web_sm")


# %% id="2GLNb1xufMqW"
def sentiment(model, sentence: str) -> bool:
    tokens = tokenizer.tokenizer(sentence)
    tokenized = [text_field.vocab.stoi[t.text] for t in tokens]
    print("|".join(text_field.vocab.itos[t] for t in tokenized))
    result = F.sigmoid(model(torch.LongTensor(tokenized).unsqueeze(1).to(device)))
    print(result.to("cpu").detach().numpy().ravel(), sentence)

sentiment(model, "it's bad")
sentiment(model, "it's good")
sentiment(model, "it's very bad")
sentiment(model, "it's very good")
sentiment(model, "it's aweful")
sentiment(model, "it's awesome")
sentiment(model, "it's great")
sentiment(model, "i like this film")
sentiment(model, "i don't like this film")
sentiment(model, "it is the best movie I have ever seen")
sentiment(model, "it is the worst movie I have ever seen")
sentiment(model, "borat was the great success, i very much liked this movie, best movie ever")
sentiment(model, "very boring movie I didn't like it")

# %% id="iwlFk4RDlpPK"
shawshank = """
Why do I want to write the 234th comment on The Shawshank Redemption? I am not sure - almost everything that could be possibly said about it has been said. But like so many other people who wrote comments, I was and am profoundly moved by this simple and eloquent depiction of hope and friendship and redemption.

The only other movie I have ever seen that effects me as strongly is To Kill a Mockingbird. Both movies leave me feeling cleaner for having watched them.
"""

sentiment(model, shawshank)

shawshank2 = """
This film is nothing but one cliche after another. Having seen many of the 100's of prison films made from the early 30's to the 50's, I was able to pull almost every minute of Shawcrap from one of those films.

While it is visually well made and acted, the story has every plot point of the "standard" prison film. They have the evil warden, innocent main character, friendly guilty guy, thugs and one good old prisoner. Don't waste your time on this one. Rent or buy some of the classic's of the prison genre
"""
sentiment(model, shawshank2)

reddit = """............ . . . . . . .. ... … ...................................................................................................."""
sentiment(model, reddit)


klasa = """
Kant has written a treatise on _The Vital Powers_; but I should like to
write a dirge on them, since their lavish use in the form of knocking,
hammering, and tumbling things about has made the whole of my life a
daily torment. Certainly there are people, nay, very many, who will
smile at this, because they are not sensitive to noise; it is precisely
these people, however, who are not sensitive to argument, thought,
poetry or art, in short, to any kind of intellectual impression: a fact
to be assigned to the coarse quality and strong texture of their brain
tissues.
"""

sentiment(model, klasa)

# %% id="e80feb3a"
sentiment(model, "great success")
# Funkcja kosztu, im bliżej 1 (target) tym funkcja kosztu maleje.

target = torch.ones([1, 1], dtype=torch.float32)  # 64 classes, batch size = 10
input_ = torch.full([1, 1], 0.1)  # A prediction (logit)

print(F.binary_cross_entropy_with_logits(input_, target))

target = torch.ones([1, 1], dtype=torch.float32)  # 64 classes, batch size = 10
input_ = torch.full([1, 1], 0.4)  # A prediction (logit)

print(F.binary_cross_entropy_with_logits(input_, target))

target = torch.ones([1, 1], dtype=torch.float32)  # 64 classes, batch size = 10
input_ = torch.full([1, 1], 0.7)  # A prediction (logit)

print(F.binary_cross_entropy_with_logits(input_, target))

target = torch.ones([1, 1], dtype=torch.float32)  # 64 classes, batch size = 10
input_ = torch.full([1, 1], 0.9)  # A prediction (logit)

print(F.binary_cross_entropy_with_logits(input_, target))

target, input_

# %% id="Axn_JzP4YkIe"

# %% id="dfac9f54"
>>> rnn = nn.RNN(3, 2, 1)
>>> input = torch.randn(5, 3, 3)
>>> h0 = torch.randn(1, 3, 2)
>>> output, hn = rnn(input, h0)
output, hn

# %% id="b8440222"
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(100, 19)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,98,1,0, 4,3,2,9],[4,3,2,9, 4,3,2,9]])
>>> embedding(input)
