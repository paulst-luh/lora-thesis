import re

import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.utils.sampling import nested_subset


# Prompting pipeline: entities are written as one "entity: TAG" line
# each and parsed back into a set of (entity, tag) pairs.

NER_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]


PROMPT_TEMPLATE = """
Identify all named entities in the text below.

Possible tags: PER, ORG, LOC, MISC

List each entity as "entity: TAG", one per line. If there are none, respond with "None".

Text:
{text}

Entities:
"""


def _extract_entities(tokens, tags):

    entities = []

    current_tokens = []
    current_type = None

    for token, tag in list(zip(tokens, tags)) + [(None, "O")]:

        if tag.startswith("B-"):

            if current_type is not None:
                entities.append((" ".join(current_tokens), current_type))

            current_tokens = [token]
            current_type = tag[2:]

        elif tag.startswith("I-") and current_type == tag[2:]:

            current_tokens.append(token)

        else:

            if current_type is not None:
                entities.append((" ".join(current_tokens), current_type))

            current_tokens = []
            current_type = None

    return entities


def format_conll2003(example):

    tags = [NER_LABELS[t] for t in example["ner_tags"]]

    entities = _extract_entities(example["tokens"], tags)

    if entities:

        target = "\n".join(
            f"{text}: {tag}" for text, tag in entities
        )

    else:

        target = "None"

    return {

        "prompt": PROMPT_TEMPLATE.format(
            text=" ".join(example["tokens"])
        ),

        "target": target,

        "label": None,

        "metadata": {}

    }


def parse_conll2003(text):

    stripped = text.strip()

    if not stripped:
        return None

    # "None" is the explicit answer for no entities: a valid empty set,
    # not a parse failure.
    if stripped.lower() == "none":
        return set()

    entities = set()

    for line in stripped.splitlines():

        match = re.match(
            r"\s*(.+?):\s*(PER|ORG|LOC|MISC)\s*$",
            line.strip()
        )

        if match:
            entities.add((match.group(1).strip(), match.group(2)))

    # No line matched "entity: TAG": a real parse failure.
    if not entities:
        return None

    return entities


def compute_conll2003_prompt_metrics(predictions, references):

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for prediction, reference in zip(predictions, references):

        true_positives += len(prediction & reference)
        false_positives += len(prediction - reference)
        false_negatives += len(reference - prediction)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {

        "eval_precision": precision,

        "eval_recall": recall,

        "eval_f1": f1

    }


def tokenize_conll2003(
    examples,
    tokenizer
):

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []

    for i in range(
        len(examples["ner_tags"])
    ):

        word_ids = tokenized.word_ids(
            batch_index=i
        )

        previous_word = None

        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(-100)

            elif word_idx != previous_word:

                label_ids.append(
                    examples["ner_tags"][i][word_idx]
                )

            else:

                label_ids.append(-100)

            previous_word = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels

    return tokenized

def prepare_conll2003_dataset(
    dataset,
    tokenizer,
    config,
    data_seed
):

    subset_size = config.get("dataset.subset_size", "full")

    train_indices = nested_subset(len(dataset["train"]), subset_size, data_seed)

    train_data = dataset["train"].select(train_indices)

    eval_data = dataset["validation"]

    train_data = train_data.map(
        lambda x:
            tokenize_conll2003(
                x,
                tokenizer
            ),
        batched=True,
        remove_columns=
            train_data.column_names
    )

    eval_data = eval_data.map(
        lambda x:
            tokenize_conll2003(
                x,
                tokenizer
            ),
        batched=True,
        remove_columns=
            eval_data.column_names
    )

    return train_data, eval_data

def compute_conll2003_metrics(eval_pred, label_names):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        pred_tags = []
        label_tags = []

        for pred, lab in zip(prediction, label):

            if lab == -100:
                continue

            pred_tags.append(label_names[int(pred)])
            label_tags.append(label_names[int(lab)])

        true_predictions.append(pred_tags)
        true_labels.append(label_tags)

    return {

        "accuracy":
            accuracy_score(
                true_labels,
                true_predictions
            ),

        "precision":
            precision_score(
                true_labels,
                true_predictions
            ),

        "recall":
            recall_score(
                true_labels,
                true_predictions
            ),

        "f1":
            f1_score(
                true_labels,
                true_predictions
            )
    }