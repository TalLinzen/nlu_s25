"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    # Data is pre-processed through the following steps:
    # 1. All input texts should be tokenized
    # 2. BERT models have a maximum input lenght, and all inputs need to be truncated to this length.
    # 3. Inputs shorter than the maximum input length should be padded to this length.
    # 4. The pre-processed inputs do not need to be in the form of PyTorch tensors.

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


    #raise NotImplementedError("Problem 1d has not been completed yet!")


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    #raise NotImplementedError("Problem 2a has not been completed yet!")
    model = BertForSequenceClassification.from_pretrained(model_name)
    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
    return model


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    #raise NotImplementedError("Problem 2b has not been completed yet!")
    #model = init_model(None, model_name, use_bitfit)

    training_args = TrainingArguments(
        output_dir = "./checkpoints",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        num_train_epochs = 4
    )

    def compute_metrics(p: EvalPrediction):
        metric = evaluate.load("accuracy")
        logits, labels = p.predictions, p.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model_init=lambda: init_model(None, model_name, use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    return trainer
    


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    #raise NotImplementedError("Problem 2c has not been completed yet!")
    search_space = {
        "learning_rate": [3e-4, 1e-4, 5e-5, 3e-5],
        "per_device_train_batch_size": [8, 16, 32, 64, 128]
    }

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", search_space["per_device_train_batch_size"]),
        }

    return {
        "direction": "maximize",
        "backend": "optuna",
        "hp_space": optuna_hp_space,
        "sampler": optuna.samplers.GridSampler(search_space)
    }


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
