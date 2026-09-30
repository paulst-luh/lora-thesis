# Datasets

The dataset files are not in git; only this README is. The runs read four
folders with `load_from_disk` (`DATA_PATHS` in `src/data/loader.py`). Any
other folder in `data/` is not used.

| Folder | Source on the Hugging Face Hub | Splits (examples) |
|---|---|---|
| `mnli/` | `nyu-mll/glue`, config `mnli` | train 392,702, validation 9,815 |
| `squad/` | `rajpurkar/squad` | train 87,599, validation 10,570 |
| `conll2003/` | `eriktks/conll2003` | train 14,041, validation 3,250, test 3,453 |
| `gsm8k/` | `openai/gsm8k`, config `main` | train 7,473, test 1,319 |

All sources are public, so no Hugging Face token is needed.

## Download

Run from the repository root with the environment active (the paths are
relative to it):

```bash
python -m scripts.download_assets
```

This also downloads the base model (see `models/README.md`) and skips
anything that already exists. To download only the datasets:

```bash
python - <<'EOF'
from datasets import DatasetDict, load_dataset

mnli = load_dataset("nyu-mll/glue", "mnli")
DatasetDict({"train": mnli["train"],
             "validation": mnli["validation_matched"]}).save_to_disk("data/mnli")
load_dataset("rajpurkar/squad").save_to_disk("data/squad")
load_dataset("eriktks/conll2003", revision="refs/convert/parquet").save_to_disk("data/conll2003")
load_dataset("openai/gsm8k", "main").save_to_disk("data/gsm8k")
EOF
```

Notes:

- MNLI keeps only `validation_matched`, saved as `validation`.
- CoNLL-2003 is read from the Hub's parquet conversion
  (`refs/convert/parquet`), because the original repository uses a loading
  script that current `datasets` versions no longer run.
- No preprocessing is needed. Each run draws its training subset
  (`dataset.subset_size`, `dataset.data_seed`) and its evaluation set
  (`dataset.eval_size`) at runtime, as set in the config.
