from transformers import AutoTokenizer, AutoModel
from pprint import pprint
# tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
# model = AutoModel.from_pretrained("avichr/heBERT")

from transformers import pipeline
from preprocess import MAX_LEN
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

hebert = pipeline(
    "fill-mask",
    model="avichr/heBERT",
    tokenizer="avichr/heBERT",
)

alephbert = pipeline(
    "fill-mask",
    model="onlplab/alephbert-base",
    # model="./alephbert/alephbert_1_epoch",
    # model="./alephbert/alephbert_40_epochs",
    # model=r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert_40_epochs",
    tokenizer="onlplab/alephbert-base",
)

scrollbert = pipeline(
    "fill-mask",
    # model="onlplab/alephbert-base",
    # model="./alephbert/alephbert_1_epoch",
    model="./alephbert/alephbert_40_epochs",
    # model=r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert_40_epochs",
    tokenizer="onlplab/alephbert-base",
)

mbert = pipeline(
    "fill-mask",
    model="bert-base-multilingual-cased",
    tokenizer="bert-base-multilingual-cased",
)

hebert.model.eval()
alephbert.model.eval()
scrollbert.model.eval()
mbert.model.eval()

# gru_model = torch.load(r"C:\Users\soki\PycharmProjects\QFIB\parameters\enc_model_bible\enc_40.pt")
gru_model = torch.load(r"C:\Users\soki\PycharmProjects\QFIB\parameters\enc_model_all_training\enc_14.pt")
# gru_model = torch.load(r"C:\Users\soki\PycharmProjects\QFIB\parameters\enc_model_all_training\enc_40.pt")

# model_names = ["heBERT", "alephBERT", "GRU"]
# token_based_model = [True, True, False]
# models = [hebert, alephbert, gru_model]

# model_names = ["heBERT", "alephBERT", "mBERT"]
# token_based_model = [True, True, True]
# models = [hebert, alephbert, mbert]

model_names = ["alephBERT", "scrollBERT"]
token_based_model = [True, True]
models = [alephbert, scrollbert]

do_sample = True
num_samples = 1000

with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data.txt", "r", encoding="utf8") as f:
# with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data_CD.txt", "r", encoding="utf8") as f:
# with open(r"C:\Users\soki\PycharmProjects\QFIB\data\full_training_set.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    if do_sample:
        lines = random.sample(lines, k=num_samples)
    
correct = [0] * len(model_names)
correct_tokens = [0] * len(model_names)

total = 0
total_tokens = 0
    
sum_word_len = 0
sum_predicted_len = 0
sum_len_diff = 0

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
    # for i in [2]:
        if token_based_model[i]:
            updated_line = line[:s] + " [MASK]" + line[e:]
            results = models[i](updated_line, top_k=10)
            result = get_best_result(results, len(y))


            # multi mask
            while len(result) < len(y):

                updated_line = (line[:s] + " ") * (s > 0) + result + "[MASK]" + line[e:]
                results = models[i](updated_line, top_k=10)
                result += get_best_result(results, len(y) - len(result))
                # result += models[i](updated_line)[0]["token_str"]
        else:
            updated_line = "s" + (line[:s] + " ") * (s > 0) + MASK_CHAR * (e-s-(s>0)) + line[e:] + "p"# * (MAX_LEN - len(line))
            # assert len(updated_line) == len(line)
            # result = result1 = predict_my_model(updated_line, models[i])
            result = result2 = predict_my_model_iterative(updated_line, models[i])
            # a=5


        if i == 1:
            sum_predicted_len += len(result)
            sum_word_len += len(y)
            sum_len_diff += abs(len(result) - len(y))

        correct[i] += sum(np.array(list(pad_prediction(result, y))) == np.array(list(y)))
    
        if result == y:
            correct_tokens[i] += 1

    total += e - s
    total_tokens += 1

for i in range(len(model_names)):
    print(f"Model: {model_names[i]}")
    print(f"char acc: {correct[i]/total}")
    print(f"token acc: {correct_tokens[i] /total_tokens}")

print(f"average prediction: {sum_predicted_len/total_tokens}")
print(f"average masked word: {sum_word_len/total_tokens}")
print(f"average abs len diff: {sum_len_diff/total_tokens}")