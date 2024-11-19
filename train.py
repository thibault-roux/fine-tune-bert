from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from datasets import Dataset
from transformers import Trainer


def get_data():
    raw_train_ds = Dataset.from_json("./data/sentiments.train.jsonlines.txt")
    raw_val_ds = Dataset.from_json("./data/sentiments.validation.jsonlines.txt")
    raw_test_ds = Dataset.from_json("./data/sentiments.test.jsonlines.txt")
    return raw_train_ds, raw_val_ds, raw_test_ds





def preprocess_function(examples):
    label = examples["score"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = label
    return examples


metric = load_metric("accuracy")
def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    raw_train_ds, raw_val_ds, raw_test_ds = get_data()


    BASE_MODEL = "camembert-base"
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 20

    # Let's name the classes 0, 1, 2, 3, 4 like their indices
    id2label = {k:k for k in range(5)}
    label2id = {k:k for k in range(5)}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

    # cleaning the full dataset
    ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}
    for split in ds:
        ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "uuid", "text", "score"])


    # Training arguments
        training_args = TrainingArguments(
        output_dir="../models/camembert-fine-tuned-regression",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    # class to train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()