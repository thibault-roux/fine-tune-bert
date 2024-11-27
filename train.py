from transformers import CamembertTokenizer
from transformers import CamembertForSequenceClassification
# import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments



def load_data(data_path="/home/ucl/cental/troux/expe/fine-tune-bert/data/Qualtrics_Annotations_formatB.csv"):
    # load csv file
    dataset = load_dataset('csv', data_files=data_path)
    return dataset




# load CamemBERT
def load_model(task, model_name):
    if task == 'classification':
        model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=4)
    elif task == 'regression':
        model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=1)
    else:
        raise ValueError('task should be either classification or regression')
    return model



# Tokenization of the full dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)




if __name__ == "__main__":
    dataset = load_data()


    # Load CamemBERT tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    # Appliquer la tokenisation au jeu de données
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load CamemBERT model
    model = load_model('classification', 'camembert-base')


    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Data split (will be modified for cross validation)
    train_valid_test_split = tokenized_datasets['train'].train_test_split(test_size=0.4)
    train_dataset = train_valid_test_split['train']
    valid_test_split = train_valid_test_split['test'].train_test_split(test_size=0.5)
    eval_dataset = valid_test_split['train']
    test_dataset = valid_test_split['test']

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    print("Results : ", metrics)


