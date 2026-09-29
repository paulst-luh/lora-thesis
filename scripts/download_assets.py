"""Download the four datasets and the base model used by the study.

Datasets are saved with `save_to_disk` to data/<name>/ (the format
src/data/loader.py reads), the model to models/qwen2-7b/. Anything that
already exists is skipped. All sources are public; no token is needed.
For gated models (e.g. the Llama/Gemma outlook configs) set your own
Hugging Face token as the environment variable HF_TOKEN beforehand.

Usage:
    python -m scripts.download_assets
"""

from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


# name -> (hub repo, config, revision, {saved split: hub split}).
# CoNLL-2003 is read from the hub's parquet conversion, since its repo
# holds a loading script that current `datasets` no longer runs.
DATASETS = {
    "mnli": ("nyu-mll/glue", "mnli", None,
             {"train": "train", "validation": "validation_matched"}),
    "squad": ("rajpurkar/squad", "plain_text", None,
              {"train": "train", "validation": "validation"}),
    "conll2003": ("eriktks/conll2003", None, "refs/convert/parquet",
                  {"train": "train", "validation": "validation", "test": "test"}),
    "gsm8k": ("openai/gsm8k", "main", None,
              {"train": "train", "test": "test"}),
}

MODEL_REPO = "Qwen/Qwen2-7B"
MODEL_DIR = Path("models/qwen2-7b")


def main():

    for name, (repo, config, revision, splits) in DATASETS.items():
        target = Path("data") / name
        if target.exists():
            print(f"{target} exists, skipped")
            continue
        loaded = load_dataset(repo, config, revision=revision)
        DatasetDict({saved: loaded[hub] for saved, hub in splits.items()}).save_to_disk(str(target))
        print(f"saved {target}")

    if MODEL_DIR.exists():
        print(f"{MODEL_DIR} exists, skipped")
    else:
        snapshot_download(MODEL_REPO, local_dir=str(MODEL_DIR))
        print(f"saved {MODEL_DIR}")


if __name__ == "__main__":
    main()
