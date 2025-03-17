"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction
import numpy as np

from train_model import preprocess_dataset




def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    #raise NotImplementedError("Problem 2b has not been completed yet!")
    model = BertForSequenceClassification.from_pretrained(directory)
    training_args = TrainingArguments(
        output_dir="./results",
        do_train = False,
        do_eval = True
    )
    def compute_metrics(p: EvalPrediction):
        metric = evaluate.load("accuracy")
        logits, labels = p.predictions, p.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    return trainer
        




if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("path_to_your_best_model")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)
