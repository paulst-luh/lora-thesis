import re

import numpy as np
from src.utils.sampling import nested_train_eval_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support
)


MNLI_LABELS = [
    "entailment",
    "neutral",
    "contradiction"
]


PROMPT_TEMPLATE = """
Determine the relationship between the premise and the hypothesis.

Possible labels:
{labels}

Premise:
{premise}

Hypothesis:
{hypothesis}

Answer:
"""


# Head pipeline

def tokenize_mnli(example, tokenizer):

    texts = [

        f"Premise: {premise}\nHypothesis: {hypothesis}"

        for premise, hypothesis in zip(
            example["premise"],
            example["hypothesis"]
        )

    ]

    tokenized = tokenizer(texts, truncation=True, max_length=512)

    tokenized["labels"] = example["label"]

    return tokenized


def prepare_mnli_dataset(dataset, tokenizer, config, data_seed):

    subset_size = config.get("dataset.subset_size", "full")

    train_indices, eval_indices = nested_train_eval_split(
        len(dataset["train"]),
        subset_size,
        data_seed,
        eval_size=config.get("dataset.eval_size", 2000)
    )

    train_data = dataset["train"].select(train_indices)
    eval_data = dataset["train"].select(eval_indices)

    train_data = train_data.map(
        lambda x: tokenize_mnli(x, tokenizer),
        batched=True
    )

    eval_data = eval_data.map(
        lambda x: tokenize_mnli(x, tokenizer),
        batched=True
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]

    train_data = train_data.remove_columns(
        [c for c in train_data.column_names if c not in keep_cols]
    )

    eval_data = eval_data.remove_columns(
        [c for c in eval_data.column_names if c not in keep_cols]
    )

    return train_data, eval_data


def compute_mnli_metrics(eval_pred, average):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    return {

        "accuracy":
            accuracy_score(labels, preds),

        "precision":
            precision_score(labels, preds, average=average, zero_division=0),

        "recall":
            recall_score(labels, preds, average=average, zero_division=0),

        "f1":
            f1_score(labels, preds, average=average, zero_division=0)
    }


def extract_mnli_confusion_matrix(predictions):

    preds = np.argmax(predictions.predictions, axis=1)

    labels = predictions.label_ids

    return confusion_matrix(labels, preds).tolist()


def extract_mnli_sample_predictions(predictions, max_examples=25):

    preds = np.argmax(predictions.predictions, axis=1)

    labels = predictions.label_ids

    examples = []

    for i in range(min(max_examples, len(preds))):

        examples.append(

            {

                "example_id":
                    int(i),

                "true_label":
                    int(labels[i]),

                "predicted_label":
                    int(preds[i]),

                "correct":
                    bool(preds[i] == labels[i])

            }

        )

    return examples


# Prompting pipeline

def format_mnli(example):

    return {

        "prompt": PROMPT_TEMPLATE.format(

            premise=example["premise"],
            hypothesis=example["hypothesis"],
            labels="\n".join(MNLI_LABELS)

        ),

        "target": MNLI_LABELS[example["label"]],

        "label": example["label"],

        "metadata": {}

    }


def _normalize(text):

    text = text.lower().strip()

    text = re.sub(r"[^\w\s]", "", text)

    return text


def parse_mnli(text):

    text = _normalize(text)

    # Exact match first.
    for label in MNLI_LABELS:

        if text == _normalize(label):

            return label

    # Otherwise, the first label contained anywhere in the text.
    for label in MNLI_LABELS:

        if _normalize(label) in text:

            return label

    return None


def compute_mnli_prompt_metrics(predictions, references, average):

    precision, recall, f1, _ = precision_recall_fscore_support(
        references,
        predictions,
        average=average,
        zero_division=0
    )

    return {

        "eval_accuracy": accuracy_score(
            references,
            predictions
        ),

        "eval_precision": precision,

        "eval_recall": recall,

        "eval_f1": f1

    }
