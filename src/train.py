from transformers import Trainer, DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForSeq2Seq

import torch

from functools import partial

from src.models.base_model import load_model
from src.models.lora_setup import apply_lora

from src.data.loader import load_dataset_by_name, DATASET_META

from src.utils.helper import get_trainable_stats
from src.utils.training import set_seed, build_training_args
from src.utils.results import save_results, save_loss_history

from src.datasets.mnli import (
    prepare_mnli_dataset,
    compute_mnli_metrics,
    compute_mnli_prompt_metrics,
    extract_mnli_sample_predictions,
    extract_mnli_confusion_matrix
)
from src.datasets.conll2003 import (
    prepare_conll2003_dataset,
    compute_conll2003_metrics,
    compute_conll2003_prompt_metrics
)
from src.datasets.squad import (
    prepare_squad_dataset,
    compute_squad_metrics,
    compute_squad_prompt_metrics
)
from src.datasets.gsm8k import compute_gsm8k_metrics

from src.prompting.prepare import prepare_prompt_train_dataset, prepare_prompt_eval_dataset
from src.prompting.runner import run_prompting, run_constrained_choice
from src.datasets import PARSERS, VERBALIZERS


# Default cap on generated tokens per dataset, sized to the expected
# answer (a single label word vs. a full chain of thought).
MAX_NEW_TOKENS = {

    "mnli": 32,

    "squad": 64,

    "conll2003": 128,

    "gsm8k": 256,

}


def train(config):

    seed = config.get("seed")

    # data_seed fixes which examples are sampled and stays constant in
    # a sweep; seed only drives training noise (init, dropout, order).
    data_seed = config["dataset.data_seed"]

    set_seed(seed)

    # Logged for RQ2 only; control flow uses dataset.name.
    task = config["task.name"]

    dataset_name = config["dataset.name"]

    pipeline = config.get("pipeline.type")

    cfg = DATASET_META[dataset_name]

    num_labels = cfg["num_labels"]

    average = cfg["average"]

    print("Loading model...")

    model, tokenizer = load_model(
        config["model.name"],
        dataset_name,
        num_labels,
        pipeline,
        dtype=config.get("model.dtype", "bfloat16")
    )

    adaptation = config["adaptation"]

    if adaptation == "lora":

        print("Applying LoRA...")

        model = apply_lora(model, config)

    elif adaptation == "fft":

        print("Running full fine-tuning (FFT)...")

    elif adaptation == "none":

        print(
            "Running zero-shot baseline "
            "(no training)..."
        )

        for p in model.parameters():
            p.requires_grad_(False)

    else:

        raise ValueError(
            f"Unknown adaptation: {adaptation}"
        )

    trainable_stats = get_trainable_stats(model)

    print("Loading dataset...")

    dataset = load_dataset_by_name(dataset_name)

    train_split = "train"

    if "validation" in dataset:
        eval_split = "validation"
    elif "test" in dataset:
        eval_split = "test"
    else:
        raise ValueError("Dataset has neither a validation nor a test split.")

    label_names = None

    if pipeline == "prompting":

        train_data = prepare_prompt_train_dataset(dataset[train_split], tokenizer, dataset_name, config, data_seed)

        eval_data = prepare_prompt_eval_dataset(dataset[eval_split], dataset_name, config)

    elif pipeline == "head":

        if dataset_name == "mnli":

            train_data, eval_data = (prepare_mnli_dataset(dataset, tokenizer, config, data_seed))

        elif dataset_name == "conll2003":

            label_names = dataset["train"].features["ner_tags"].feature.names

            train_data, eval_data = (prepare_conll2003_dataset(dataset, tokenizer, config, data_seed))

        elif dataset_name == "squad":

            train_data, eval_data, eval_features, eval_examples = (prepare_squad_dataset(dataset, tokenizer, config, data_seed))

    metrics_fn = None

    if pipeline == "head":

        if dataset_name == "mnli":

            metrics_fn = partial(
                compute_mnli_metrics,
                average=average
            )

        elif dataset_name == "conll2003":

            metrics_fn = partial(
                compute_conll2003_metrics,
                label_names=label_names
            )

        elif dataset_name == "squad":

            metrics_fn = None

    if pipeline == "prompting":

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    elif dataset_name == "conll2003":

        data_collator = DataCollatorForTokenClassification(tokenizer)

    else:

        data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(

        model=model,

        args=build_training_args(config,seed),

        train_dataset=train_data,

        eval_dataset=eval_data,

        data_collator=data_collator,

        compute_metrics=metrics_fn
    )

    train_result = None

    torch.cuda.reset_peak_memory_stats()

    if adaptation == "none":

        print("Skipping training (zero-shot baseline)...")

    else:

        print("Starting training...")

        train_result = trainer.train()

    if pipeline == "head":

        metrics = trainer.evaluate()

        # Head runs produce no free text, so there is nothing to parse.
        metrics["parse_failure_rate"] = None

    elif pipeline == "prompting":

        parser = PARSERS[dataset_name]

        max_length = config.get("dataset.max_seq_length", 512)

        if cfg["task"] == "classification":

            # Score the label verbalizers directly instead of parsing
            # free text, so no run looks worse only because its output
            # could not be parsed.
            predictions = run_constrained_choice(
                model,
                tokenizer,
                eval_data,
                choices=VERBALIZERS[dataset_name],
                max_length=max_length
            )

            parse_failure_rate = 0.0

        else:

            max_new_tokens = config.get("eval.max_new_tokens", MAX_NEW_TOKENS.get(dataset_name, 32))

            do_sample = config.get("eval.do_sample", False)

            raw_predictions = run_prompting(
                model,
                tokenizer,
                eval_data,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                do_sample=do_sample
            )

            predictions = [

                parser(pred)

                for pred in raw_predictions

            ]

            parse_failures = sum(

                1
                for pred in predictions
                if pred is None

            )

            parse_failure_rate = parse_failures / len(predictions) if predictions else 0.0

            # Replace parser failures (None, already counted above) with
            # an empty prediction the metric functions can score.
            empty_prediction = set() if dataset_name == "conll2003" else ""

            predictions = [

                pred if pred is not None else empty_prediction

                for pred in predictions

            ]

        references = [

            sample["target"]

            for sample in eval_data

        ]

        if dataset_name == "mnli":

            metrics = compute_mnli_prompt_metrics(

                predictions,
                references,
                average=average

            )

        elif dataset_name == "squad":

            metrics = compute_squad_prompt_metrics(

                predictions,
                references

            )

        elif dataset_name == "conll2003":

            references = [parser(ref) for ref in references]

            metrics = compute_conll2003_prompt_metrics(

                predictions,
                references

            )

        elif dataset_name == "gsm8k":

            references = [parser(ref) for ref in references]

            metrics = compute_gsm8k_metrics(

                predictions,
                references

            )

        metrics["parse_failure_rate"] = parse_failure_rate

    if pipeline == "head" and dataset_name == "squad":

        predictions = trainer.predict(eval_data)

        squad_metrics = compute_squad_metrics(predictions, eval_examples, eval_features)

        metrics.update(squad_metrics)

    metrics["runtime_seconds"] = None

    for entry in reversed(trainer.state.log_history):

        if "train_runtime" in entry:

            metrics["runtime_seconds"] = entry["train_runtime"]

            break

    metrics["num_train_epochs"] = (trainer.state.epoch)

    # Realized step count, so iso_step can be checked per run.
    metrics["optimizer_steps"] = (trainer.state.global_step)

    metrics["train_loss"] = (train_result.training_loss if train_result is not None else None)

    metrics["peak_allocated_vram_gb"] = (torch.cuda.max_memory_allocated() / 1024**3)

    metrics["peak_reserved_vram_gb"] = (torch.cuda.max_memory_reserved() / 1024**3)

    loss_history = (trainer.state.log_history)

    save_loss_history(config, loss_history)

    sample_predictions = []
    confusion_matrix_data = []

    if pipeline == "head" and dataset_name == "mnli":

        predictions = trainer.predict(eval_data)

        sample_predictions = extract_mnli_sample_predictions(predictions)

        confusion_matrix_data = extract_mnli_confusion_matrix(predictions)

    save_results(config, metrics, trainable_stats, sample_predictions, confusion_matrix_data)
