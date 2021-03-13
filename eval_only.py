import pandas as pd
from sklearn.model_selection import train_test_split
from main_batch import evaluate, DATASET_PATH, VOCAB_PATH
import joblib
import torch
import os

print_sentences = True
dataset = pd.read_json(DATASET_PATH)
dataset = dataset.sample(frac=0.1)
train_sentences, dev_sentences = train_test_split(dataset)
vocab = joblib.load(VOCAB_PATH)

#load model
postfix=""
enc = torch.load(os.path.join("parameters", f'enc_40{postfix}.pt'))
dec = torch.load(os.path.join("parameters", f'dec_40{postfix}.pt'))

train_accuracy = evaluate(train_sentences, enc, dec, print_sentences)
dev_accuracy = evaluate(dev_sentences, enc, dec, print_sentences)

print(f"train accuracy: {train_accuracy}")
print(f"dev accuracy: {dev_accuracy}")
