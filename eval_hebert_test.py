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

from preprocess import MASK_CHAR
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


hebert = pipeline(
    "fill-mask",
    model="avichr/heBERT",
    tokenizer="avichr/heBERT"
)

alephbert = pipeline(
    "fill-mask",
    model="onlplab/alephbert-base",
    tokenizer="onlplab/alephbert-base"
)

gru_model = torch.load(r"C:\Users\soki\PycharmProjects\QFIB\parameters\enc_model_bible\enc_40.pt")

model_names = ["heBERT", "alephBERT", "GRU"]
token_based_model = [True, True, False]
models = [hebert, alephbert, gru_model]


with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data.txt", "r", encoding="utf8") as f:
    lines = f.readlines()#[:100]
    
correct = [0] * len(model_names)
correct_tokens = [0] * len(model_names)

total = 0
total_tokens = 0
    

    
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

    for i in range(len(model_names)):
        if token_based_model[i]:
            updated_line = line[:s] + " [MASK]" + line[e:]
            result = models[i](updated_line)[0]["token_str"]
        else:
            updated_line = line[:s] + " " + MASK_CHAR * (e-s-1) + line[e:]
            assert len(updated_line) == len(line)
            result = predict_my_model(updated_line, models[i])
    

        correct[i] += sum(np.array(list(pad_prediction(result, y))) == np.array(list(y)))
    
        if result == y:
            correct_tokens[i] += 1

    total += e - s
    total_tokens += 1

for i in range(len(model_names)):
    print(f"Model: {model_names[i]}")
    print(f"char acc: {correct[i]/total}")
    print(f"token acc: {correct_tokens[i] /total_tokens}")
