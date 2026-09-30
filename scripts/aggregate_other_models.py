"""Aggregate the small-model pilot runs into one CSV.

Covers results/test/ (gemma-3-1b, gpt2, llama-3.2-1b, qwen3-0.6b).
Unlike src/utils/aggregate_runs.py it adds a "model" column, since
these runs vary the base model. build_flat_row() is reused unchanged,
so the main study's output is not affected.

Usage:
    python -m scripts.aggregate_other_models [--results-root DIR]
                                             [--out FILE]
"""

import argparse
import glob
import json
import os

import pandas as pd

from src.utils.results import build_flat_row


def find_run_dirs(results_root):
    """Yield every directory with a metrics/ folder of JSON results."""
    for dirpath, _, filenames in os.walk(results_root):

        if os.path.basename(dirpath) != "metrics":
            continue

        if not any(f.endswith(".json") for f in filenames):
            continue

        yield os.path.dirname(dirpath)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/test")
    parser.add_argument("--out", default="results/other_models_runs.csv")
    args = parser.parse_args()

    rows = []
    n_files = 0

    for run_dir in sorted(find_run_dirs(args.results_root)):

        metrics_dir = os.path.join(run_dir, "metrics")

        for filename in sorted(os.listdir(metrics_dir)):

            if not filename.endswith(".json"):
                continue

            path = os.path.join(metrics_dir, filename)
            n_files += 1

            with open(path) as f:
                data = json.load(f)

            row = build_flat_row(data["config"], data["metrics"], data["trainable"])

            # The column build_flat_row() lacks, e.g. "gemma-3-1b"
            # (without the "models/" prefix).
            model_name = data["config"].get("model.name")
            row["model"] = (
                model_name.removeprefix("models/") if model_name else None
            )

            row["experiment_name"] = data.get("experiment_name")
            row["experiment_group"] = data.get("experiment_group")
            row["source_file"] = path

            rows.append(row)

    assert rows, f"no metrics/*.json files found under {args.results_root}"

    out = pd.DataFrame(rows)

    # Fail before writing if any row lacks a model name.
    assert out["model"].notna().all(), (
        "some row(s) have no model.name in their config -- "
        f"{out['model'].isna().sum()} row(s) affected"
    )

    models_found = sorted(out["model"].unique())
    print(f"models found: {models_found}")
    print(f"rows per model:\n{out['model'].value_counts()}")

    out.to_csv(args.out, index=False)

    print(f"\nread {n_files} result file(s), wrote {len(out)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
