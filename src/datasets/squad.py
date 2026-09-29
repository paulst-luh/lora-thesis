import collections
import json
import logging
import os
import re
import string
from collections import Counter
from typing import Optional, Tuple
from evaluate import load

import numpy as np
from tqdm.auto import tqdm

from src.utils.sampling import nested_train_eval_split


logger = logging.getLogger(__name__)


# Prompting pipeline

PROMPT_TEMPLATE = """
Answer the question using the shortest possible span of text taken directly from the context.

Context:
{context}

Question:
{question}

Answer:
"""


def format_squad(example):

    return {

        "prompt": PROMPT_TEMPLATE.format(

            context=example["context"],
            question=example["question"]

        ),

        "target": example["answers"]["text"][0],

        "label": None,

        "metadata": {}

    }


def parse_squad(text):

    # Keep the first line only, since a short span is asked for. An
    # empty generation counts as a parse failure (None).
    return text.strip().splitlines()[0].strip() if text.strip() else None


def _normalize_answer(text):

    text = text.lower()

    text = "".join(ch for ch in text if ch not in string.punctuation)

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = " ".join(text.split())

    return text


def _exact_match_score(prediction, reference):

    return int(_normalize_answer(prediction) == _normalize_answer(reference))


def _f1_score(prediction, reference):

    pred_tokens = _normalize_answer(prediction).split()

    ref_tokens = _normalize_answer(reference).split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return int(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_squad_prompt_metrics(predictions, references):

    exact_match = sum(

        _exact_match_score(pred, ref)

        for pred, ref in zip(predictions, references)

    ) / len(predictions)

    f1 = sum(

        _f1_score(pred, ref)

        for pred, ref in zip(predictions, references)

    ) / len(predictions)

    return {

        "eval_exact_match": exact_match,

        "eval_f1": f1

    }


def tokenize_squad(examples, tokenizer):

    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
        padding=False,
        stride=128,
        return_overflowing_tokens=True
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    example_ids = []

    for sample_idx in sample_mapping:

        example_ids.append(examples["id"][sample_idx])

    tokenized["example_id"] = example_ids

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):

        sample_idx = sample_mapping[i]

        answer = examples["answers"][sample_idx]

        answer_start = answer["answer_start"][0]

        answer_text = answer["text"][0]

        answer_end = (answer_start + len(answer_text))

        sequence_ids = tokenized.sequence_ids(i)

        context_start = None
        context_end = None

        for idx, seq_id in enumerate(sequence_ids):

            if seq_id == 1:

                if context_start is None:
                    context_start = idx

                context_end = idx

        if (context_start is None or context_end is None):

            start_positions.append(0)
            end_positions.append(0)

            continue

        start_token = context_start

        if (offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end):

            start_positions.append(0)
            end_positions.append(0)

            continue

        while (start_token <= context_end and offsets[start_token][0] <= answer_start):
            start_token += 1

        start_positions.append(start_token - 1)

        end_token = context_end

        while (end_token >= context_start and offsets[end_token][1] >= answer_end):
            end_token -= 1

        end_positions.append(end_token + 1)

    tokenized["start_positions"] = (start_positions)

    tokenized["end_positions"] = (end_positions)

    return tokenized

def prepare_squad_dataset(dataset, tokenizer, config, data_seed):

    subset_size = config.get("dataset.subset_size", "full")

    train_indices, eval_indices = nested_train_eval_split(
        len(dataset["train"]),
        subset_size,
        data_seed,
        eval_size=config.get("dataset.eval_size", 2000)
    )

    train_data = dataset["train"].select(train_indices)
    eval_examples = dataset["train"].select(eval_indices)

    train_data = train_data.map(
        lambda x: tokenize_squad(x, tokenizer),
        batched=True,
        remove_columns=train_data.column_names
    )

    train_data = train_data.remove_columns(["offset_mapping", "example_id"])

    eval_features = eval_examples.map(
        lambda x: tokenize_squad(x, tokenizer),
        batched=True,
        remove_columns=eval_examples.column_names
    )

    eval_dataset = eval_features.remove_columns(["offset_mapping", "example_id"])

    return (train_data, eval_dataset, eval_features, eval_examples)

def compute_squad_metrics(predictions, eval_examples, eval_features):

    squad_metric = load("squad")

    predictions = postprocess_squad_predictions(

        examples=eval_examples,

        features=eval_features,

        predictions=predictions.predictions
    )

    formatted_predictions = [

        {

            "id": k,

            "prediction_text": v

        }

        for k, v in predictions.items()

    ]

    references = [

        {

            "id": ex["id"],

            "answers": ex["answers"]

        }

        for ex in eval_examples

    ]

    return squad_metric.compute(

        predictions=formatted_predictions,

        references=references

    )
    

def postprocess_squad_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """Turn start/end logits into answer strings from the contexts.

    Adapted from the Hugging Face question-answering example
    (utils_qa.py), for models that return start and end logits.

    Args:
        examples: The raw (non-tokenized) examples.
        features: The tokenized features of `examples`.
        predictions: Tuple (start_logits, end_logits), each with one
            row per feature.
        version_2_with_negative: Whether examples may have no answer.
        n_best_size: Number of best start/end logits to combine.
        max_answer_length: Maximum answer length in tokens.
        null_score_diff_threshold: Pick the null answer if the best
            score is below the null score minus this threshold (only
            with version_2_with_negative).
        output_dir: If given, save the predictions there as JSON.
        prefix: Optional prefix for the saved file names.
        log_level: Logging level.

    Returns:
        dict: Example id -> predicted answer text.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Map each example to its features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # Output dictionaries.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        # Indices of the features of the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # Maps logit positions to character spans in the context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # If given, used to skip answers without maximum context.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Try all pairs of the n_best_size highest start/end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip indices out of bounds or outside the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Skip spans that are negative or too long.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Skip answers without maximum context, if known.
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Re-add the null prediction if its low score dropped it.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Gather the answer text from the context via the offsets.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # Rare edge case: no non-null prediction, so add a dummy one.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Numerically stable softmax over the scores.
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Find the best non-empty prediction first.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Compare it with the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Cast numpy floats to float for JSON.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # Optionally save all dictionaries to output_dir.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions