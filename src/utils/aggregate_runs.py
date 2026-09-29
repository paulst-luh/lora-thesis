"""Aggregate all run results into one validated dataframe.

Each run writes its own metrics/result_<id>.json. This script works in
two stages:

1. build_runs_csv_files(): rebuild runs.csv in every run directory
   from its JSON files via build_flat_row(), so it always reflects the
   current schema. Runs offline in a single process, because the
   cluster filesystem rejects flock() under concurrent access.
2. aggregate_and_validate(): concatenate all runs.csv files and fail
   on any factor-column NaN not explained by CONDITIONAL_NA.

Usage:
    python -m src.utils.aggregate_runs
    python -m src.utils.aggregate_runs --out results/runs_all.csv
"""

import argparse
import glob
import json
import os

import pandas as pd

from src.utils.results import build_flat_row


FACTOR_COLUMNS = [
    "dataset",
    "pipeline",
    "adaptation",
    "lora_target",
    "rank",
    "alpha",
    "scaling_ratio",
    "learning_rate",
    "subset_size",
    "regime",
    "training_regime",
    "seed",
    "metric",
    "parse_failure_rate",
    "runtime_s",
    "peak_mem_gb",
    "trainable_params",
    "optimizer_steps",
]

# Column -> row mask under which a NaN in that column is expected.
CONDITIONAL_NA = {

    "lora_target": lambda df: df["adaptation"] != "lora",
    "rank": lambda df: df["adaptation"] != "lora",
    "alpha": lambda df: df["adaptation"] != "lora",
    "scaling_ratio": lambda df: df["adaptation"] != "lora",

    "parse_failure_rate": lambda df: df["pipeline"] != "prompting",

    "runtime_s": lambda df: df["adaptation"] == "none",

}


def _is_legacy_row(row):
    """Return True if `row` predates the current result schema.

    That is, a factor column is None without a CONDITIONAL_NA reason.
    """
    for column in FACTOR_COLUMNS:

        if row.get(column) is not None:
            continue

        if column in CONDITIONAL_NA and CONDITIONAL_NA[column](row):
            continue

        return True

    return False


def find_run_dirs(results_root):
    """Yield every directory with a metrics/ folder of JSON results."""
    for dirpath, _, filenames in os.walk(results_root):

        if os.path.basename(dirpath) != "metrics":
            continue

        if not any(f.endswith(".json") for f in filenames):
            continue

        yield os.path.dirname(dirpath)


def build_runs_csv_files(results_root):
    """Rebuild runs.csv in every run directory under results_root."""
    written = []

    for run_dir in find_run_dirs(results_root):

        metrics_dir = os.path.join(run_dir, "metrics")

        rows = []

        n_legacy = 0

        for filename in sorted(os.listdir(metrics_dir)):

            if not filename.endswith(".json"):
                continue

            path = os.path.join(metrics_dir, filename)

            with open(path) as f:
                data = json.load(f)

            row = build_flat_row(
                data["config"],
                data["metrics"],
                data["trainable"]
            )

            # Skip rather than backfill: legacy rows predate the current
            # pipeline and cannot be reconstructed honestly.
            if _is_legacy_row(row):

                n_legacy += 1

                continue

            row["experiment_name"] = data.get("experiment_name")
            row["experiment_group"] = data.get("experiment_group")
            row["source_file"] = path

            rows.append(row)

        if n_legacy:
            print(f"  skipping {n_legacy} pre-schema legacy result(s) in {run_dir}")

        out_path = os.path.join(run_dir, "runs.csv")

        if not rows:

            # No current rows here: remove any stale runs.csv.
            if os.path.exists(out_path):
                os.remove(out_path)

            continue

        pd.DataFrame(rows).to_csv(out_path, index=False)

        written.append(out_path)

    return written


def aggregate_and_validate(results_root):

    paths = sorted(glob.glob(os.path.join(results_root, "**", "runs.csv"), recursive=True))

    if not paths:

        raise FileNotFoundError(
            f"No runs.csv found under {results_root}."
        )

    combined = pd.concat(
        [pd.read_csv(p) for p in paths],
        ignore_index=True
    )

    print(f"Loaded {len(paths)} runs.csv file(s), {len(combined)} row(s) total.\n")

    failures = {}

    for column in FACTOR_COLUMNS:

        if column not in combined.columns:

            failures[column] = "column missing entirely from the combined dataframe"

            continue

        is_na = combined[column].isna()

        if column in CONDITIONAL_NA:

            is_na = is_na & ~CONDITIONAL_NA[column](combined)

        if is_na.any():

            failures[column] = combined.loc[is_na, "source_file"].tolist()

    if failures:

        print("FAIL -- unexplained NaNs found:\n")

        for column, detail in failures.items():

            if isinstance(detail, str):

                print(f"  {column}: {detail}")

            else:

                print(f"  {column}: {len(detail)} row(s), e.g.")

                for source in detail[:5]:
                    print(f"    {source}")

        raise ValueError(
            "Factor-column NaN validation failed -- see above."
        )

    print("OK -- no unexplained NaNs in any factor column.")

    for column, condition in CONDITIONAL_NA.items():

        if column not in combined.columns:
            continue

        n = (combined[column].isna() & condition(combined)).sum()

        if n:
            print(f"  ({column}: {n} row(s) legitimately NaN for their adaptation/pipeline)")

    return combined


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results-root", default="results")

    parser.add_argument("--out", default=None, help="Optional path to also save the combined dataframe as one CSV.")

    args = parser.parse_args()

    written = build_runs_csv_files(args.results_root)

    print(f"\n(Re)generated {len(written)} runs.csv file(s) under {args.results_root}.\n")

    if not written:

        print(
            "Nothing to aggregate yet -- no run under "
            f"{args.results_root} matches the current schema "
            "(every existing result predates it and was skipped above)."
        )

        return

    combined = aggregate_and_validate(args.results_root)

    if args.out:

        combined.to_csv(args.out, index=False)

        print(f"\nSaved combined dataframe to {args.out}")


if __name__ == "__main__":
    main()
