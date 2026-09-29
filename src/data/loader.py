from datasets import load_from_disk


DATASET_META = {

    "mnli": {
        "name": "MNLI",
        "classes": 3,
        "task": "classification",
        "difficulty": "large",
        "num_labels": 3,
        "average": "macro"
    },

    "squad": {
        "name": "SQuAD",
        "classes": None,
        "task": "qa",
        "difficulty": "medium",
        "num_labels": 2,
        "average": None
    },

    "conll2003": {
        "name": "CoNLL-2003",
        "classes": 9,
        "task": "token_classification",
        "difficulty": "medium",
        "num_labels": 9,
        "average": None
    },

    "gsm8k": {
        "name": "GSM8K",
        "classes": None,
        "task": "reasoning",
        "difficulty": "medium",
        "num_labels": None,
        "average": None
    }
}


DATA_PATHS = {

    "mnli": "data/mnli",

    "squad": "data/squad",

    "conll2003": "data/conll2003",

    "gsm8k": "data/gsm8k",

}


def load_dataset_by_name(dataset_name):

    if dataset_name not in DATA_PATHS:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    return load_from_disk(DATA_PATHS[dataset_name])