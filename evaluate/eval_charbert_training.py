import transformers
from transformers import pipeline
import torch
from transformers.tokenization_utils import TruncationStrategy
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    without_mask = query.replace('[MASK]', 'א') #replace with random char that will be replaced carfully with mask
    tokenized = do_tokenize(without_mask)
    tokenized['input_ids'][0][mask_idx + 1] = pipe.tokenizer.mask_token_id
    return pipe(tokenized, tokenized=True, top_k=top_k)

def pad_prediction(pred, target):
    pred = pred[:len(target)]
    if len(pred) < len(target):
        pred = pred + "p" * (len(target) - len(pred))
    return pred

def unmask_multi(query, top_k=5):
    nonspaces = query.replace(' ','')
    mask_indices = []
    tmp_nonspaces = nonspaces
    while "[MASK]" in tmp_nonspaces:
        mask_indices += [tmp_nonspaces.find("[MASK]")]
        tmp_nonspaces = tmp_nonspaces.replace("[MASK]", "א", 1)

    without_mask = query.replace('[MASK]', 'א') #replace with random char that will be replaced carfully with mask
    tokenized = do_tokenize(without_mask)

    prob = 1
    for i in range(len(mask_indices)):
        tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.mask_token_id
        res = pipe(tokenized, tokenized=True, top_k=top_k)
        token = res[0]["token_str"]
        prob *= res[0]["score"]
        tokenized['input_ids'][0][mask_indices[i] + 1] = pipe.tokenizer.convert_tokens_to_ids([token])[0]

    return pipe.tokenizer.convert_tokens_to_string(pipe.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0], skip_special_tokens=True)), prob


random.seed(42)
num_samples = 100
do_sample = True
# with open(r"./data/test_data.txt", "r", encoding="utf8") as f:
with open(r"./data/full_training_set.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    if do_sample:
        lines = random.sample(lines, k=num_samples)

char_acc = []
token_acc = []


ITERS = 16
for i in range(1, ITERS + 1):
    pipe = pipeline(
        "fill-mask",
        model=f"./char-base-hebrew-bert/char-bert/checkpoint-{i}0000",
        tokenizer=r"char-base-hebrew-bert/model"
    )

    pipe._parse_and_tokenize = _parse_and_tokenize
    pipe.model.eval()


    correct = 0
    correct_tokens = 0
    total = 0
    total_tokens = 0


    for line in tqdm(lines):

        start_mask = random.randint(0, len(line) - 1)
        while line[start_mask] == " ":
            start_mask = random.randint(0, len(line) - 1)

        s = e = start_mask
        while s > 0 and line[s] != " ":
            s -= 1
        while e < len(line) and line[e] != " ":
            e += 1
        y = line[s + 1:e]

        updated_line = (line[:s] + " ") * (s>0) + "[MASK]" * (e-s-(s>0)) + line[e:]
        result = unmask_multi(updated_line)[0][s+1:e]

        correct += sum(np.array(list(pad_prediction(result, y))) == np.array(list(y)))
        if result == y:
            correct_tokens += 1

        total += e - s
        total_tokens += 1


    print(f"Model: Char model")
    print(f"char acc: {correct/total}")
    print(f"token acc: {correct_tokens /total_tokens}")
    char_acc += [correct / total]
    token_acc += [correct_tokens / total_tokens]


plt.plot(range(1, ITERS + 1), char_acc)
plt.title("Character accuracy")
plt.ylabel("char_acc")
plt.xlabel("x*10000 samples")
plt.savefig(f"./char-base-hebrew-bert/char-bert/train_char_acc.png")
plt.clf()

plt.plot(range(1, ITERS + 1), token_acc)
plt.title("Token accuracy")
plt.ylabel("token_acc")
plt.xlabel("x*10000 samples")
plt.savefig(f"./char-base-hebrew-bert/char-bert/train_token_acc.png")

