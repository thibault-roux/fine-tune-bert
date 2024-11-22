from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
# from datasets import load_metric # deprecated
import evaluate
from transformers import TrainingArguments
from datasets import Dataset
from transformers import Trainer
# split dataset
from sklearn.model_selection import train_test_split
import pandas as pd


cache_path = "/home/ucl/cental/troux/expe/bert/models/hfcache"


def get_data(namefile):
    # load csv
    data = Dataset.from_csv(namefile)
    # only keep columns "text_indice", "text" and "gold_score_20"
    data = data.select_columns(["text_indice", "text", "gold_score_20"])
    
    # Convert to pandas DataFrame
    df = data.to_pandas()
    
    # split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    
    # Convert back to Dataset
    train_ds = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
    val_ds = Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"])
    test_ds = Dataset.from_pandas(test_df).remove_columns(["__index_level_0__"])
    
    # print(train_ds)
    # exit()
    return train_ds, val_ds, test_ds




def preprocess_function(examples):
    label = examples["gold_score_20"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = label
    return examples


# metric = load_metric("accuracy")
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    namefile = "data/Qualtrics_Annotations_corrige.csv"
    train_ds, val_ds, test_ds = get_data(namefile)


    BASE_MODEL = "camembert-base"
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 20

    # Let's name the classes 0, 1, 2, 3, 4 like their indices
    id2label = {k:k for k in range(5)}
    label2id = {k:k for k in range(5)}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=cache_path)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id, cache_dir=cache_path)

    # cleaning the full dataset
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}
    for split in ds:
        ds[split] = ds[split].map(preprocess_function, remove_columns=["text_indice", "text", "gold_score_20"])


    # Training arguments
        training_args = TrainingArguments(
        output_dir="./models/camembert-fine-tuned-regression",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
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