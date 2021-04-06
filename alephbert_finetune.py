from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="onlplab/alephbert-base",
    tokenizer="onlplab/alephbert-base"
)

from transformers import LineByLineTextDataset
tokenizer = fill_mask.tokenizer
model = fill_mask.model
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/full_training_set.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./alephbert",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./alephbert")