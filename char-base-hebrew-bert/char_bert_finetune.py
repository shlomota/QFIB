from transformers import pipeline
import logging
from transformers import LineByLineTextDataset

# fill_mask = pipeline(
#     "fill-mask",
#     model="onlplab/alephbert-base",
#     tokenizer="onlplab/alephbert-base"
# )
fill_mask = pipeline(
    "fill-mask",
    model="./char-base-hebrew-bert/model/",
    tokenizer="./char-base-hebrew-bert/model"
)


tokenizer = fill_mask.tokenizer
model = fill_mask.model
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/full_training_set.txt",
    block_size=128,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/test_data.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./char-base-hebrew-bert/char-bert",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_gpu_train_batch_size=16,
    save_steps=10_000,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./char-base-hebrew-bert/char-bert")
tokenizer.save_pretrained("./char-bert")

logging.info("Started evaluation on train!")
metrics = trainer.evaluate(eval_dataset=dataset)
logging.info(metrics)
print(metrics)

logging.info("Started evaluation on test!")
metrics = trainer.evaluate(eval_dataset=eval_dataset)
logging.info(metrics)
print(metrics)

