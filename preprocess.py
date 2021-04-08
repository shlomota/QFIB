from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os
from vocabulary import Vocabulary

MASK_CHAR = "_"
PAD_CHAR = "p"
SOS_CHAR = "s"
UNK_CHAR = "u"
MAX_LEN = 256
MAX_MASK_LEN = 40

BASE_PATH = "data"
RAW_DATA_FILENAME = "data.txt"
# BASE_PATH = "data/he"
RAW_DATA_FILENAME = "wikidata.txt"
RAW_DATA_FILENAME = "full_training_set.txt"
DATASET_PATH = os.path.join(BASE_PATH, "dataset.json")
VOCAB_PATH = os.path.join(BASE_PATH, "vocab.pkl")
MASK_RATIO = 0.15
# MAX_MASK = 5

UNIFORM_MASK = False

def generate_dataset():
    with open(os.path.join(BASE_PATH, RAW_DATA_FILENAME), "r", encoding="utf8") as f:
        sents = f.readlines()

    #add sos char
    sents = [SOS_CHAR + sent for sent in sents]
    padded = [sent[:MAX_LEN] + PAD_CHAR*(MAX_LEN - len(sent)) for sent in sents]
    lengths = [len(sent) for sent in padded]
    assert np.all(np.array(lengths) == MAX_LEN)


    # just used to make dict
    if os.path.isfile(VOCAB_PATH):
        vocab = joblib.load(VOCAB_PATH)
    else:
        cv = CountVectorizer(analyzer="char", encoding="utf8")
        cv.fit(sents+[MASK_CHAR + PAD_CHAR + UNK_CHAR])
        vocab = Vocabulary(cv.vocabulary_)
        joblib.dump(vocab, VOCAB_PATH)

    processed = np.array([[vocab.w2i[char] for char in sent] for sent in padded])




    #print(max(map(lambda x: len(x), processed)))
    #236 is max len


    dataset = pd.DataFrame(columns=["x", "y"])

    for processed_sent in tqdm(processed):
        # mask_len = random.randint(1, min(len(processed_sent), MAX_MASK))
        sent_len = list(processed_sent).index(vocab.w2i[PAD_CHAR]) if vocab.w2i[PAD_CHAR] in list(processed_sent) else len(processed_sent)
        mask_len = int(MASK_RATIO * sent_len)
        start_mask = random.randint(0, sent_len - mask_len)
        #masking
        if UNIFORM_MASK:
            y = np.full(MAX_MASK_LEN, vocab.w2i[PAD_CHAR])
            y[:mask_len] = processed_sent[start_mask:start_mask + mask_len].copy()
        else:
            y = processed_sent[start_mask:start_mask + mask_len].copy()
        x = processed_sent.copy()
        x[start_mask:start_mask + mask_len] = [vocab.w2i[MASK_CHAR] for i in range(mask_len)]
        dataset.loc[len(dataset)] = [x, y]

    print(dataset.head())
    dataset.to_json(DATASET_PATH)
    # dataset.to_csv(DATASET_PATH)
    """
    sentences: 23213
    distinct words: 39538
    """

if __name__ == "__main__":
    generate_dataset()