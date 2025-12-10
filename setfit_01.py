'''testing python setfit library'''
from datasets import load_dataset # type: ignore
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset # type: ignore

ds = load_dataset("SetFit/ag_news")
train_data = ds['train']
train_set, validation_set = train_data.train_test_split(test_size=0.2, seed=42).values()
train_set = sample_dataset(train_set, num_samples=100)
test_set = ds['test']

del ds, train_data


args = TrainingArguments(
    output_dir="results",
    batch_size=4,
    num_epochs=1,
)

# We now have a train_set and a validation_set
# We need to create a model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

# We need to train the model
trainer = Trainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=validation_set,
    args=args,
)

trainer.train()

# Evaluating
metrics = trainer.evaluate(test_set)
print(metrics)


model.save_pretrained("setfit-ag-news-20251209")
