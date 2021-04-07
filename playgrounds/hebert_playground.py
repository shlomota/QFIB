from transformers import AutoTokenizer, AutoModel
from pprint import pprint
# tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
# model = AutoModel.from_pretrained("avichr/heBERT")

from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="avichr/heBERT",
    tokenizer="avichr/heBERT"
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
# pprint(fill_mask("הקורונה לקחה את [MASK] ולנו לא נשאר דבר."))
# pprint(fill_mask("בראשית ברא [MASK] את השמים ואת"))
pprint(fill_mask("בראשית בר[MASK] אלהים את השמים ואת"))
# pprint(fill_mask("ואלה שמות בני [MASK] הבאים מצרימה."))
# pprint(fill_mask("ואלה שמות בני [MASK] הבאים [MASK] את יעקב איש וביתו באו."))
# pprint(fill_multi("ואלה שמות בני [MASK] הבאים [MASK] את יעקב איש וביתו באו."))
# pprint(fill_mask("בראשית ברא [MASK] את [MASK] ואת הארץ."))


# much worse
# fill_mask = pipeline('fill-mask', model='bert-base-multilingual-cased', tokenizer="bert-base-multilingual-cased")
# # pprint(fill_mask("Hello I'm a [MASK] model."))
# pprint(fill_mask("בראשית ברא [MASK] את השמים ואת הארץ."))