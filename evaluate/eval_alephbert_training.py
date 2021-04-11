import os
from transformers import AutoTokenizer, AutoModel
from pprint import pprint
# tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
# model = AutoModel.from_pretrained("avichr/heBERT")

from transformers import pipeline
from tqdm import tqdm
import random
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

# from preprocess import MAX_LEN
# from preprocess import MASK_CHAR
MASK_CHAR = "_"
DEVICE = "cuda:0"

def fill_multi(text, fill_mask):
    prob = 1
    while "MASK" in text:
        text_li = text.split("[MASK]")
        txt = "".join(text_li[:1]) + "[MASK]" + "[UNK]".join(text_li[1:])
        fill_result = fill_mask(txt)
        prob *= fill_result[0]["score"]
        text = "".join(text_li[:1]) + fill_result[0]["token_str"] + "[MASK]".join(text_li[1:])
    return text, prob

def pad_prediction(pred, target):
    pred = pred[:len(target)]
    if len(pred) < len(target):
        pred = pred + "p" * (len(target) - len(pred))
    return pred

def predict_my_model(sent, model):
    input_tensor = torch.tensor([model.vocab.w2i[char] for char in sent]).to(DEVICE)
    encoder_outputs, encoder_h_m = model(input_tensor.unsqueeze(0))
    preds = encoder_outputs.squeeze()[torch.where(input_tensor==model.vocab.w2i[MASK_CHAR])]
    preds = torch.argmax(preds, dim=-1)
    return "".join([model.vocab.i2w[index] for index in preds.tolist()])

def predict_my_model_iterative(sent, model):
    input_tensor = torch.tensor([model.vocab.w2i[char] for char in sent]).to(DEVICE)
    preds = []
    while model.vocab.w2i[MASK_CHAR] in input_tensor:
        encoder_outputs, encoder_h_m = model(input_tensor.unsqueeze(0))
        index = torch.where(input_tensor==model.vocab.w2i[MASK_CHAR])[0][0]
        pred = encoder_outputs.squeeze()[index]
        pred = torch.argmax(pred, dim=-1)
        preds += [pred.item()]
        input_tensor[index] = pred.item()
    return "".join([model.vocab.i2w[index] for index in preds])

def get_best_result(results, len_y):
    for res in results:
        if len(res["token_str"]) == len_y:
            return res["token_str"]
    return results[0]["token_str"]


random.seed(42)
do_sample = True
num_samples = 100

char_acc = []
token_acc = []

#Read data
# with open(r"C:\Users\soki\PycharmProjects\QFIB\data\full_training_set.txt", "r", encoding="utf8") as f:
# with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data.txt", "r", encoding="utf8") as f:
# with open(r"./data/full_training_set.txt", "r", encoding="utf8") as f:
with open(r"./data/test_data.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    if do_sample:
        lines = random.sample(lines, k=num_samples)


ITERS = 15
for i in range(1, ITERS + 1):
    alephbert = pipeline(
        "fill-mask",
        # model="onlplab/alephbert-base",
        # model="./alephbert/alephbert_1_epoch",
        model=f"./alephbert/alephbert/checkpoint-{i}0000",
        # model=r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert_40_epochs",
        tokenizer="onlplab/alephbert-base",
    )

    alephbert.model.eval()
    token_based_model = True
    
    correct = 0
    correct_tokens = 0

    total = 0
    total_tokens = 0

    sum_word_len = 0
    sum_predicted_len = 0
    sum_len_diff = 0

    random.seed(42)
    for line in tqdm(lines):
        # mask_len = random.randint(1, min(len(processed_sent), MAX_MASK))
        # mask_len = int(MASK_RATIO * len(line))

        start_mask = random.randint(0, len(line) - 1)
        while line[start_mask] == " ":
            start_mask = random.randint(0, len(line) - 1)

        s = e = start_mask
        # e = start_mask
        while s > 0 and line[s] != " ":
            s -= 1
        while e < len(line) and line[e] != " ":
            e += 1
        y = line[s + 1:e]

        if token_based_model:
            updated_line = line[:s] + " [MASK]" + line[e:]
            results = alephbert(updated_line, top_k=10)
            result = get_best_result(results, len(y))


            # multi mask
            while len(result) < len(y):

                updated_line = (line[:s] + " ") * (s > 0) + result + "[MASK]" + line[e:]
                results = alephbert(updated_line, top_k=10)
                result += get_best_result(results, len(y) - len(result))
        else:
            updated_line = "s" + (line[:s] + " ") * (s > 0) + MASK_CHAR * (e-s-(s>0)) + line[e:] + "p"# * (MAX_LEN - len(line))
            result = result2 = predict_my_model_iterative(updated_line, alephbert.model)


        correct += sum(np.array(list(pad_prediction(result, y))) == np.array(list(y)))

        if result == y:
            correct_tokens += 1

    total += e - s
    total_tokens += 1


    print(f"samples viewed: {i}0000") #4880 ratio
    print(f"char acc: {correct/total}")
    print(f"token acc: {correct_tokens /total_tokens}")
    char_acc += [correct / total]
    token_acc += [correct_tokens / total_tokens]

plt.plot(range(1, ITERS + 1), char_acc)
plt.title("Character accuracy")
plt.ylabel("char_acc")
plt.xlabel("x*10000 samples")
plt.savefig(f"./alephbert/alephbert/char_acc.png")
plt.clf()

plt.plot(range(1, ITERS + 1), token_acc)
plt.title("Token accuracy")
plt.ylabel("token_acc")
plt.xlabel("x*10000 samples")
plt.savefig(f"./alephbert/alephbert/token_acc.png")