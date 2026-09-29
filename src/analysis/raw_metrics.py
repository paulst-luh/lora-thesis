"""Read the sub-metrics stored only in each run's result JSON.

Precision, recall, exact match etc. are in the raw `metrics` dict but
not in the aggregated CSV. Used by appendix figure A2. GSM8K has no
sub-metrics (exact match is its only score).
"""

import glob
import json
import os

import pandas as pd


RESULTS_ROOT = "results"

SUB_METRIC_KEYS = {
    "mnli": ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"],
    "conll2003": ["eval_precision", "eval_recall", "eval_f1"],
    "squad": ["eval_exact_match", "eval_f1"],
}

# Block -> directory suffixes in results/<dataset>/prompting_<suffix>/.
# The ratio-1 completion runs belong to their main-sweep block, as in
# scripts/build_final_csv.py.
BLOCK_DIRS = {
    "zeroshot": ["zeroshot"],
    "fft": ["fft"],
    "main_qv": ["main_qv", "main_qv_ratio1_extend"],
    "main_attention": ["main_attention", "main_attention_ratio1_extend"],
    "main_full": ["main_full", "main_full_ratio1_extend"],
}


def load_sub_metrics(dataset, blocks, results_root=RESULTS_ROOT):
    """Return one row per run with its factors and sub-metrics.

    Factors come from each JSON's flattened `row`, sub-metrics from its
    raw `metrics` dict. `source_file` is kept for traceability.
    """
    if dataset not in SUB_METRIC_KEYS:
        raise ValueError(f"{dataset!r} has no sub-metrics defined (GSM8K never will -- see module docstring)")

    keys = SUB_METRIC_KEYS[dataset]
    rows = []

    for block in blocks:

        paths = []
        for block_dir in BLOCK_DIRS[block]:
            pattern = os.path.join(results_root, dataset, f"prompting_{block_dir}", "metrics", "*.json")
            paths += sorted(glob.glob(pattern))

        for path in paths:

            with open(path) as f:
                data = json.load(f)

            row = data["row"]
            metrics = data["metrics"]

            record = {
                "dataset": dataset,
                "block": block,
                "seed": row.get("seed"),
                "lora_target": row.get("lora_target"),
                "rank": row.get("rank"),
                "alpha": row.get("alpha"),
                "learning_rate": row.get("learning_rate"),
                "source_file": path,
            }

            for key in keys:
                record[key] = metrics.get(key)

            rows.append(record)

    return pd.DataFrame(rows)
