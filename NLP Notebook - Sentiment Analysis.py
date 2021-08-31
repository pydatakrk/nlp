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
text_field = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", fix_length=100)
label_field = data.LabelField(dtype=torch.float) # torch.float because GPUs use floats

# Load dataset and split to train and test data
# IMDB dataset (about movies)
train, test = datasets.IMDB.splits(text_field=text_field, label_field=label_field)

# %% id="8dc7770b"
# Split to train and validation set - 80% to train_set, 20% to validation_set
# The original set is 25k descriptions(?) so train_set after the split is 20k and valid_set is 5k.
train_set, valid_set = train.split(0.8)
len(train_set), len(valid_set)  # 20_000, 5_000
text_field.build_vocab(train_set, max_size=25_000)
label_field.build_vocab(train_set)

assert len(text_field.vocab) == 25_002

# %% id="2d2abd5f"
# Map int to string and string to int
# text_field.vocab.itos[186] -> 'though'
# text_field.vocab.stoi['though'] -> 186

# %% id="6cc86523" colab={"base_uri": "https://localhost:8080/"} outputId="18d914d0-0a1e-4da4-b706-0039bfe6dddd"
text_field.vocab.itos[:10]

# %% id="089e9316"
len(max(train_set, key=lambda x: len(x.text)).text)

# but we can do better!
train_buckets, valid_buckets, test_buckets = data.BucketIterator.splits(
    (train_set, valid_set, test), batch_size=64, device=device
)

# %% id="35a768f0"
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


class NLPModuleLSTM(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_size, out_features):
        # before parent
        super().__init__()
        # after parent
        # warstwa osadzeń/osadzanie(?) embedding
        # wektory w przestrzeni znaczeniowej słów
        self.embedding = nn.Embedding(num_embedding, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, 2)
        self.linear = nn.Linear(hidden_size * 2, out_features)
        self.dropout = nn.Dropout()

    def forward(self, input):
        embed_output = self.embedding(input)
        lstm_output, (hidden_output1, hidden_output2) = self.lstm(embed_output)
        drop_output = self.dropout(
            torch.cat((hidden_output1[-2, :, :], hidden_output1[-1, :, :]), dim=1)
        )
        lin_output = self.linear(drop_output)

        return lin_output


# %% id="45972d1f"
num_embedding = len(text_field.vocab)
embedding_dim = 100
hidden_size = 256
out_features = 1

# num_embedding, embedding_dim, hidden_size, out_features

model = NLPModuleLSTM(num_embedding, embedding_dim, hidden_size, out_features)


# %% id="a3c6bc10" colab={"base_uri": "https://localhost:8080/"} outputId="f63742c2-8d02-4000-b5fc-7dbb5e3c55f2"
def policz(mod):
    return sum(p.numel() for p in mod.parameters())


policz(model)

# %% id="3fe5b41c"
# Stochastic gradient descent SGD
# minimalizować funkcję kosztu (szukanie minimum)

import torch.optim as optim

optimiser = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

# %% id="8058738f" colab={"base_uri": "https://localhost:8080/"} outputId="674d4f22-48e4-4475-d7a8-6e39ac3ddd73"
ciretrion = criterion.to(device)
model = model.to(device)


def binary_accuracy(prediction, target):
    prediction = F.sigmoid(prediction)
    prediction = torch.round(prediction)

    compared = (prediction == target).float()
    return torch.mean(compared)


T = torch.tensor
binary_accuracy(T([0, 0.5, 0.2, 0.001, 0.8]), T([0, 1, 1, 1, 1]))

# %% id="21dc8651"
import numpy as np
import tqdm


def train(mod, data, optimiser, criterion):
    losses = []
    metrics = []

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


# %% id="SDhHPvnNbKnz" colab={"base_uri": "https://localhost:8080/"} outputId="49534f35-db09-4d6d-c4ab-0a88fe319bf8"
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


# sentiment(model, "it's bad")
# sentiment(model, "it's good")
# sentiment(model, "it's very bad")
# sentiment(model, "it's very good")
# sentiment(model, "it's aweful")
# sentiment(model, "it's awesome")
# sentiment(model, "it's great")
# sentiment(model, "i like this film")
# sentiment(model, "i don't like this film")
# sentiment(model, "it is the best movie I have ever seen")
# sentiment(model, "it is the worst movie I have ever seen")
# sentiment(model, "borat was the great success, i very much liked this movie, best movie ever")
# sentiment(model, "very boring movie I didn't like it")

# %% colab={"base_uri": "https://localhost:8080/"} id="iwlFk4RDlpPK" outputId="9a36c9e8-be74-4a83-e0e9-d95b25ace78d"
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


schopenhauer = """
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

sentiment(model, schopenhauer)

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
