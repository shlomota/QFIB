#! pip install tokenizers

from pathlib import Path
import transformers
#
from tokenizers import CharBPETokenizer
from tokenizers import ByteLevelBPETokenizer, Word
# from tokenizers.models import
#
paths = [str(x) for x in Path("data/he").glob("**/*.txt")]
#
# # Initialize a tokenizer
# tokenizer = CharBPETokenizer()
tokenizer = ByteLevelBPETokenizer()
#
# # Customize training
tokenizer.train(files=paths, vocab_size=0, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
#
# # Save files to disk
tokenizer.save_model("hebert2")




tokenizer = ByteLevelBPETokenizer(
    "hebert2/vocab.json",
    "hebert2/merges.txt",
)
# # tokenizer._tokenizer.post_processor = BertProcessing(
# #     ("</s>", tokenizer.token_to_id("</s>")),
# #     ("<s>", tokenizer.token_to_id("<s>")),
# # )
# tokenizer.enable_truncation(max_length=512)
#
# print(
#     tokenizer.encode("שלום עליכם").tokens
# )
#
#
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=40,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
#
from transformers import RobertaTokenizerFast, RobertaTokenizer

tokenizer = RobertaTokenizerFast.from_pretrained("./hebert2", max_len=512)
tokenizer = RobertaTokenizer.from_pretrained("./hebert2", max_len=512)
#
#
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

# model.num_parameters()
#
#
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/he/wikidata.txt",
    block_size=128,
)

a=5
#
#
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
#

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./hebert2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
#
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./hebert2")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./hebert2",
    tokenizer="./hebert2"
)

print(fill_mask("בראשית ברא אלהי<mask>."))