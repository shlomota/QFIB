from transformers import AutoTokenizer, AutoModel
from pprint import pprint
# tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
# model = AutoModel.from_pretrained("avichr/heBERT")

from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="bert-base-multilingual-cased",
    tokenizer="bert-base-multilingual-cased"
)

fill_mask = pipeline(
    "fill-mask",
    "bert-base-multilingual-cased",
)
def fill_multi(text):
    prob = 1
    while "MASK" in text:
        text_li = text.split("[MASK]")
        txt = "".join(text_li[:1]) + "[MASK]" + "[UNK]".join(text_li[1:])
        fill_result = fill_mask(txt)
        prob *= fill_result[0]["score"]
        text = "".join(text_li[:1]) + fill_result[0]["token_str"] + "[MASK]".join(text_li[1:])
    return text, prob


import transformers
from transformers import pipeline
import torch
from transformers.tokenization_utils import TruncationStrategy


pipe = pipeline("fill-mask", r"..\char-base-hebrew-bert\model")#, tokenizer="./model")
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



# pprint(fill_mask("הקורונה לקחה את [MASK] ולנו לא נשאר דבר."))
# pprint(unmask("בראשית ברא [MASK]להים את השמים ואת"))
text = "בנימין נתניהו נולד ב[MASK][MASK]ראל"
# pprint(unmask_multi("בראשית ברא [MASK][MASK]הים את השמים ואת"))
pprint(unmask_multi(text))
print(pipe.model.num_parameters())
#
# text = "בראשית בר[MASK] אלהים את השמים ואת הארץ"
# pprint(fill_mask(text))
# pprint(fill_mask("בראשית ברא [MASK]להים את השמים ואת"))
# pprint(fill_mask("ואלה שמות בני [MASK] הבאים מצרימה."))
# pprint(fill_mask("ואלה שמות בני [MASK] הבאים [MASK] את יעקב איש וביתו באו."))
# pprint(fill_multi("ואלה שמות בני [MASK] הבאים [MASK] את יעקב איש וביתו באו."))
# pprint(fill_mask("בראשית ברא [MASK] את [MASK] ואת הארץ."))

# print(fill_multi("בראשית ברא [MASK] [MASK] [MASK] [MASK] [MASK] את השמים ואת"))
# print(fill_multi("בראשית ברא [MASK] [MASK] [MASK] [MASK] [MASK] את השמים ואת"))

# much worse
# fill_mask = pipeline('fill-mask', model='bert-base-multilingual-cased', tokenizer="bert-base-multilingual-cased")
# # pprint(fill_mask("Hello I'm a [MASK] model."))
# pprint(fill_mask("בראשית ברא [MASK] את השמים ואת הארץ."))

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained("bert-base-multilingual-cased")
# # text = "Replace me by any text you'd like."
# text = "בראשית"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(encoded_input)
# # print(output.shape)