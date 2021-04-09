import transformers
from transformers import pipeline
import torch
from transformers.tokenization_utils import TruncationStrategy
import random
import numpy as np
from tqdm import tqdm



# pipe = pipeline("fill-mask", r"..\char-base-hebrew-bert\model")#, tokenizer="./model")
# pipe = pipeline(
#     "fill-mask",
#     model="bert-base-multilingual-cased",
#     tokenizer="bert-base-multilingual-cased"
# )

pipe = pipeline(
    "fill-mask",
    model="char-base-hebrew-bert/model",
    tokenizer=r"char-base-hebrew-bert/model"
)
# print(fill_mask.tokenizer.tokenize("שלום עליכם"))

def fill_multi(text):
    prob = 1
    while "MASK" in text:
        text_li = text.split("[MASK]")
        txt = "".join(text_li[:1]) + "[MASK]" + "[UNK]".join(text_li[1:])
        fill_result = unmask(txt)
        prob *= fill_result[0]["score"]
        text = "".join(text_li[:1]) + fill_result[0]["token_str"][-1] + "[MASK]".join(text_li[1:])
    return text, prob


def do_tokenize(inputs):
    return pipe.tokenizer(
        inputs,
        add_special_tokens=True,
        return_tensors=pipe.framework,
        padding=True,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
    )

def _parse_and_tokenize(
        inputs, tokenized=False, **kwargs
):
    if not tokenized:
        inputs = do_tokenize(inputs)
    return inputs


def unmask(query, top_k=5):
    nonspaces = query.replace(' ','')
    mask_idx = nonspaces.find('[MASK]')
    without_mask = query.replace('[MASK]', 'ץ') #replace with random char that will be replaced carfully with mask
    tokenized = do_tokenize(without_mask)
    tokenized['input_ids'][0][mask_idx + 1] = pipe.tokenizer.mask_token_id
    return pipe(tokenized, tokenized=True, top_k=top_k)


import re
# def my_pipe()
def unmask_multi(query, top_k=5):
    nonspaces = query.replace(' ','')
    mask_indices = []
    tmp_nonspaces = nonspaces
    while "[MASK]" in tmp_nonspaces:
        mask_indices += [tmp_nonspaces.find("[MASK]")]
        tmp_nonspaces = tmp_nonspaces.replace("[MASK]", "א", 1)
    # mask_indices = [m.start() for m in re.finditer("\[MASK\]", nonspaces)]
    # mask_idx = nonspaces.find('[MASK]')
    without_mask = query.replace('[MASK]', 'א') #replace with random char that will be replaced carfully with mask
    tokenized = do_tokenize(without_mask)

    #
    # for i in range(len(mask_indices)):
    #     # tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.unk_token_id
    #     tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.mask_token_id
    # mask_indices =
    prob = 1
    for i in range(len(mask_indices)):
        tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.mask_token_id
        res = pipe(tokenized, tokenized=True, top_k=top_k)
        token = res[0]["token_str"]
        prob *= res[0]["score"]
        tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.convert_tokens_to_ids([token])[0]

    return pipe.tokenizer.convert_tokens_to_string(pipe.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0], skip_special_tokens=True)), prob



pipe._parse_and_tokenize = _parse_and_tokenize

# tokenizer = transformers.ConvBertTokenizer.from_pretrained(r"C:\Users\soki\PycharmProjects\char-base-hebrew-bert\model")
# model = transformers.ConvBertModel.from_pretrained(r"C:\Users\soki\PycharmProjects\char-base-hebrew-bert\model")
# text = "שלום לכם ילדים ויל[MASK]ות"
# text = "שלום ע[MASK]לם"
# text = "שלום עלי[MASK]ם"
# text = "בנימין נתניהו נולד ב[MASK]שר[MASK]ל"
# text = "בנימין נתניהו נולד ב[MASK][MASK]ראל"
# text = "בנימין נתניהו נולד ביש[MASK]אל"
# text = "בנימין נתניהו נולד ב[MASK][MASK]ריקה"

# print(unmask(text))
# print(fill_multi(text))
# print(unmask_multi(text))
# tokens = tokenizer.tokenize(text)
# input = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
# inputs = tokenizer(text, return_tensors="pt")
# a = model(**inputs)
# hidden = a["last_hidden_state"]
# results = torch.softmax(a["last_hidden_state"], dim=2).squeeze()[16, :]


# b=5
# print(tokenizer.tokenize("שלום עליכם"))
# text = "שלום לכם ילדים ויל[MASK]ות"
# print(pipe(text)[0])
# text = "Happy birthday to y[MASK]u"
# print(pipe(text)[0])

# with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data.txt", "r", encoding="utf8") as f:
with open(r"C:\Users\soki\PycharmProjects\QFIB\data\full_training_set.txt", "r", encoding="utf8") as f:
    lines = f.readlines()[:100]

correct = 0
correct_tokens = 0
total = 0
total_tokens = 0

def pad_prediction(pred, target):
    pred = pred[:len(target)]
    if len(pred) < len(target):
        pred = pred + "p" * (len(target) - len(pred))
    return pred


for line in tqdm(lines):
    # mask_len = random.randint(1, min(len(processed_sent), MAX_MASK))
    # mask_len = int(MASK_RATIO * len(line))

    start_mask = random.randint(0, len(line) - 1)
    while line[start_mask] == " ":
        start_mask = random.randint(0, len(line) - 1)

    s = e = start_mask
    # e = start_mask
    # while s > 0 and line[s] != " ":
    #     s -= 1
    # while e < len(line) and line[e] != " ":
    #     e += 1
    y = line[s + 1:e]

    updated_line = (line[:s] + " ") * (s>0) + "[MASK]" * (e-s-(s>0)) + line[e:]
    # assert len(updated_line) == len(line)
    result = unmask_multi(updated_line)[0][s+1:e]

    correct += sum(np.array(list(pad_prediction(result, y))) == np.array(list(y)))
    if result == y:
        correct_tokens += 1

    total += e - s
    total_tokens += 1

print(f"Model: Char model")
print(f"char acc: {correct/total}")
print(f"token acc: {correct_tokens /total_tokens}")



"""
Output:
{'sequence': 'שלום לכם ילדים ויל " ות', 'score': 0.34216684103012085, 'token': 13, 'token_str': '"'}
{'sequence': 'happy birthday to yo u', 'score': 0.5778210163116455, 'token': 553, 'token_str': '##o'}
"""
