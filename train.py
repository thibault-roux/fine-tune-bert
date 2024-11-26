from transformers import CamembertTokenizer
from transformers import CamembertForSequenceClassification
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



label_to_id = {
            'Très Facile': 0,
            'Facile': 1,
            'Accessible': 2,
            '+Complexe': 3
        }

id_to_label = {
    0: 'Très Facile',
    1: 'Facile',
    2: 'Accessible',
    3: '+Complexe'
}

def load_data(data_path, task):
    # load csv file
    dataset = load_dataset('csv', data_files=data_path)
    if task == 'classification':
        label_column_name = 'gold_score_20_label'
        dataset = dataset.map(lambda examples: {'labels': label_to_id[examples[label_column_name]]})
    elif task == 'regression':
        label_column_name = 'gold_score_20'
        dataset = dataset.map(lambda examples: {'labels': float(examples[label_column_name])})
    else:
        raise ValueError('task should be either classification or regression')
    return dataset # dataset contains only a train but it will be splitted later




# load CamemBERT
def load_model(task, model_name):
    if task == 'classification':
        model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=4)
    elif task == 'regression':
        model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=1)
    else:
        raise ValueError('task should be either classification or regression')
    return model



def stratified_split(dataset, stratify_column='gold_score_20_label', test_size=0.4):
    """
    Perform a stratified split on the dataset based on the labels.
    Returns train, validation, and test datasets.
    """
    # Convert to pandas dataframe for stratified split
    df = pd.DataFrame(dataset["train"])

    # Split into train (60%) and temporary set (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_column],  # Stratify by label
        random_state=42
    )

    # Split temporary set into validation (20%) and test (20%)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[stratify_column],
        random_state=42
    )

    # Convert pandas dataframes back to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Return dataset splits as a DatasetDict
    return DatasetDict({'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})



# Tokenization of the full dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)





metric = evaluate.load("accuracy")
def compute_metrics(eval_pred): # for classification
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # print only the first 10 examples
        for i in range(10):
            print("True:", labels[i], "- Predictions:", predictions[i], "- Logits:", logits[i])
        return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    task = 'classification'
    # task = 'regression'
    data_path = '/home/ucl/cental/troux/expe/fine-tune-bert/data/Qualtrics_Annotations_formatB.csv'
    model_name = 'camembert-base'
    # model_name = 'camembert/camembert-large'


    # Load CamemBERT model
    model = load_model(task, model_name)


    # Load data
    dataset = load_data(data_path, task)
    # Load CamemBERT tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    # Apply tokenization to dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Stratified split
    split_datasets = stratified_split(tokenized_datasets)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['eval']
    test_dataset = split_datasets['test']


    """
    TO DELETE AFTER DEBUGGING
    """
    # for each set, print the distributions of the labels in percentage
    for dataset in [train_dataset, eval_dataset, test_dataset]:
        # print name of the dataset
        if dataset == train_dataset:
            print("Train dataset")
        elif dataset == eval_dataset:
            print("Eval dataset")
        else:
            print("Test dataset")
        print("Dataset size : ", len(dataset))
        for i in range(4):
            print(id_to_label[i], ":", len([x for x in dataset['labels'] if x == i]) / len(dataset) * 100, "%")
        print()
    input()




    # batch_size = 16
    batch_size = 64
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./models",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="loss", # instead of accuracy
        load_best_model_at_end=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        # save_total_limit=2,
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluation on validation
    metrics = trainer.evaluate()
    print("Validation Results : ", metrics)

    # Evaluation on test
    metrics = trainer.evaluate(test_dataset)
    print("Test Results : ", metrics)

    # print prediction of the first 10 examples
    predictions = trainer.predict(test_dataset)
    for i in range(10):
        print("Predicted label :", str(np.argmax(predictions.predictions[i])), "- True label : ", str(test_dataset['labels'][i]), "\tPrediction :", str(predictions.predictions[i]))
        # predictions.predictions[i] = [-0.19786438  0.15751775  0.16081156 -0.12675774] as a numpy array
        # print the max value of the array
        print("Max value : ", )



