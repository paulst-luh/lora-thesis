from src.datasets import build_prompt_examples
from src.utils.sampling import nested_subset, fixed_eval_subset


def tokenize_prompt(
    example,
    tokenizer,
    max_length,
    loss_on_completion=True
):

    prompt = example["prompt"]
    target = example["target"]

    full_text = prompt + " " + target + tokenizer.eos_token

    full = tokenizer(

        full_text,

        truncation=True,
        padding=False,
        max_length=max_length

    )

    labels = full["input_ids"].copy()

    if loss_on_completion:

        # Mask the prompt tokens. This assumes the prompt tokenizes the
        # same alone as inside the full text, which holds because every
        # template ends in a newline before the separator space.
        prompt_length = len(

            tokenizer(
                prompt,
                truncation=True,
                max_length=max_length
            )["input_ids"]

        )

        labels[:prompt_length] = [-100] * prompt_length

    return {

        "input_ids": full["input_ids"],

        "attention_mask": full["attention_mask"],

        "labels": labels

    }


def subset_dataset(dataset, config, data_seed):

    subset_size = config.get("dataset.subset_size", "full")

    indices = nested_subset(len(dataset), subset_size, data_seed)

    return dataset.select(indices)


def prepare_prompt_train_dataset(
    dataset,
    tokenizer,
    dataset_name,
    config,
    data_seed
):

    loss_on_completion = config.get("training.loss_on_completion", True)

    max_length = config.get("dataset.max_seq_length", 512)

    dataset = subset_dataset(dataset, config, data_seed)

    dataset = build_prompt_examples(
        dataset,
        dataset_name
    )

    dataset = dataset.map(
        lambda example: tokenize_prompt(
            example,
            tokenizer,
            max_length,
            loss_on_completion
        ),
        remove_columns=dataset.column_names
    )

    return dataset

def prepare_prompt_eval_dataset(
    dataset,
    dataset_name,
    config
):

    # Independent of data_seed and subset_size, so every run is scored
    # on the same examples (see fixed_eval_subset).
    eval_size = config.get("dataset.eval_size", 2000)

    indices = fixed_eval_subset(len(dataset), eval_size)

    dataset = dataset.select(indices)

    dataset = build_prompt_examples(
        dataset,
        dataset_name
    )

    return dataset.select_columns([
        "prompt",
        "target"
    ])