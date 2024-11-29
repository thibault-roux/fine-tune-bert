from transformers import CamembertTokenizer
from transformers import CamembertForSequenceClassification
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer



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
        min_value = 1
        max_value = 20
        # dataset = dataset.map(lambda examples: {'labels': float(examples[label_column_name])})
        dataset = dataset.map(lambda examples: {'labels': (float(examples[label_column_name]) - min_value) / (max_value - min_value)})
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

# convert regression predictions to the correct scale between min and max
def convert_regression_predictions(predictions, min_value=1, max_value=20):
    return predictions * (max_value - min_value) + min_value

# convert regression predictions to the correct scale between min and max


# evaluate on classification
metric_classification = evaluate.load("accuracy")
def compute_metrics_classification(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # print only the first 10 examples
        for i in range(20):
            print("True:", labels[i], "- Predictions:", predictions[i], "- Logits:", logits[i])
        return metric_classification.compute(predictions=predictions, references=labels)

# evaluate on regression
metric_regression = evaluate.load("spearmanr")
def compute_metrics_regression(eval_pred):
        logits, labels = eval_pred
        # print only the first 10 examples
        for i in range(20):
            print("True:", convert_regression_predictions(labels[i]), "- Predictions:", convert_regression_predictions(logits[i]))
        return metric_regression.compute(predictions=logits, references=labels)


# Custom trainer to overwrite the loss function
class CustomClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Custom loss function for training.
        By default, Trainer uses CrossEntropyLoss for classification.
        This can be overridden here for custom loss.
        """
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)
        logits = outputs.logits

        # distributions of classes in the training dataset
        distributions = [4.693, 45.848, 40.072, 9.386]
        # inverse of the distributions
        inverse_distributions = [1/x for x in distributions]
        # normalize the inverse distributions
        inverse_distributions = [x/sum(inverse_distributions) for x in inverse_distributions] # [0.5815731457118071, 0.05952981095850442, 0.06811047047378495, 0.29078657285590354]

        # Custom loss function:
        class_weights = torch.tensor([inverse_distributions]).to(logits.device)  # tensor([[0.5816, 0.0595, 0.0681, 0.2908]], device='cuda:0')
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.squeeze())
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
class CustomWeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # Compute weights inversely proportional to target frequency or values
        weights = torch.where(targets > 10, torch.tensor(2.0), torch.tensor(1.0)).to(predictions.device)
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        return weighted_mse.mean()
    
class CustomRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Custom loss function for training.
        By default, Trainer uses MSE for regression.
        This can be overridden here for custom loss.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Custom weighted loss
        loss_fn = CustomWeightedMSELoss()
        loss = loss_fn(logits.squeeze(), labels)

        return (loss, outputs) if return_outputs else loss



if __name__ == "__main__":
    task = 'classification'
    # task = 'regression'
    data_path = '/home/ucl/cental/troux/expe/fine-tune-bert/data/Qualtrics_Annotations_formatB.csv'
    # model_name = 'camembert-base' # 0.6344086021505376
    model_name = 'camembert/camembert-large'
    # model_name = 'almanach/camembertv2-base' # 0.7741935483870968 avec 64 batch size and 0.7096774193548387 avec 16 batch size


    # Load CamemBERT model
    model = load_model(task, model_name)


    # Load data
    dataset = load_data(data_path, task)
    # Load CamemBERT tokenizer
    # tokenizer = CamembertTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Apply tokenization to dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Stratified split
    split_datasets = stratified_split(tokenized_datasets)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['eval']
    test_dataset = split_datasets['test']

    # normalize for regression
    if task == 'regression':
        train_labels = train_dataset


    # List all columns except the ones you want to keep
    columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]

    # Remove the unnecessary columns from each split
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    eval_dataset = eval_dataset.remove_columns(columns_to_remove)
    test_dataset = test_dataset.remove_columns(columns_to_remove)



    batch_size = 16
    # batch_size = 64
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./models",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
        metric_for_best_model="loss", # instead of accuracy
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100, # 30
        weight_decay=0.01,
    )

    # Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )
    if task == 'classification':
        print("Train dataset:", train_dataset)
        trainer = CustomClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_classification,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
    elif task == 'regression':
        trainer = CustomRegressionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_regression,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
    else:
        raise ValueError('task should be either classification or regression')



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
    for i in range(20):
        if task == 'classification':
            print("Predicted label :", str(np.argmax(predictions.predictions[i])), "- True label : ", str(test_dataset['labels'][i]), "\tPrediction :", str(predictions.predictions[i]))
        elif task == 'regression':
            print("Predicted score :", str(convert_regression_predictions(predictions.predictions[i])), "- True score : ", str(convert_regression_predictions(test_dataset['labels'][i])))
        else:
            raise ValueError('task should be either classification or regression')