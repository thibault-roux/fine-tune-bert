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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math



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
    # Change this to real number
    examples["label"] = float(label)
    return examples


"""
# metric = load_metric("accuracy")
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred): # for classification
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
"""

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

def regression_to_classification_metric(eval_pred): # evaluate the regression as a classification
    logits, labels = eval_pred
    predictions = np.round(logits)
    # convert predictions with this rule : 0 to 5 -> 0, 6 to 10 -> 1, 11 to 15 -> 2, 16 to 20 -> 3
    predictions = np.where(predictions < 6, 0, np.where(predictions < 11, 1, np.where(predictions < 16, 2, 3)))
    labels = np.where(labels < 6, 0, np.where(labels < 11, 1, np.where(labels < 16, 2, 3)))
    return {"accuracy": np.mean(predictions == labels)}

    



if __name__ == "__main__":
    namefile = "data/Qualtrics_Annotations_corrige.csv"
    train_ds, val_ds, test_ds = get_data(namefile)


    BASE_MODEL = "camembert-base"
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 20


    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=cache_path)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1, cache_dir=cache_path)
    
    # cleaning the full dataset
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}
    for split in ds:
        ds[split] = ds[split].map(preprocess_function, remove_columns=["text_indice", "text", "gold_score_20"]) # only keep input_ids, attention_mask and label (i.e. gold_score_20 as float)



    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, tight_layout=True)
    distributions = []

    axs[0].set_title("Train"); axs[1].set_title("Validation"); axs[2].set_title("Test"); 
    train_distributions = axs[0].hist(train_ds["score"], bins=5)
    val_distributions = axs[1].hist(val_ds["score"], bins=5)
    test_distributions = axs[2].hist(test_ds["score"], bins=5)

    # convert each element of distributions with the rule 1 to 5 -> 0, 6 to 10 -> 1, 11 to 15 -> 2, 16 to 20 -> 3

    for distributions, ax in zip([train_distributions, val_distributions, test_distributions], axs):
        for j in range(5):
            # Display the counts on each column of the histograms
            ax.text(distributions[1][j], distributions[0][j], str(int(distributions[0][j])), weight="bold")

    plt.show()
    # save
    plt.savefig("distributions.png")
    plt.close()
    # exit()
            

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
        compute_metrics=regression_to_classification_metric,
    )

    trainer.train()

    # load the best model and evaluate it on the test set
    print("trainer.evaluate(ds['test']):", trainer.evaluate(ds["test"]))


    nb_batches = math.ceil(len(test_ds)/BATCH_SIZE)
    y_preds = []

    for i in range(nb_batches):
        input_texts = test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["text"]
        input_labels = test_ds[i * BATCH_SIZE: (i+1) * BATCH_SIZE]["gold_score_20"]
        encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
        y_preds += model(**encoded).logits.reshape(-1).tolist()

    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame([test_ds["text"], test_ds["gold_score_20"], y_preds], ["Text", "Score", "Prediction"]).T
    df["Rounded Prediction"] = df["Prediction"].apply(round)
    print(df)
    # incorrect_cases = df[df["Score"] != df["Rounded Prediction"]]
    # print("incorrect_cases:")
    # print(incorrect_cases)
    