"""Shared data loading and the single collapse rule for all figures.

See docs/plot_plan.md §0 for the collapse rule and
docs/figure_data_audit.md for the contents of the results CSV.
"""

from pathlib import Path

import pandas as pd


# The only place a results path may be hard-coded.
RESULTS_CSV = Path("results/prompting_study_final.csv")

# One configuration ("cell"): runs in the same cell differ only by
# seed. is_collapsed() and aggregate_cells() both group by this.
CELL_COLS = [
    "dataset",
    "block",
    "lora_target",
    "rank",
    "alpha",
    "learning_rate",
    "subset_size",
    "training_regime",
]


def load_runs(path=RESULTS_CSV):
    """Load the aggregated prompting-study CSV.

    Fails loudly if the file is missing instead of falling back to
    anything; the error message explains how to rebuild it.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Regenerate it first:\n"
            "  python -m src.utils.aggregate_runs --results-root results "
            "--out results/runs_prompting_study.csv\n"
            "then rebuild results/prompting_study_final.csv per "
            "docs/experimental_design_context.md §3.8."
        )

    return pd.read_csv(path)


def attach_strategy(df):
    """Return a copy of `df` with a readable `strategy` column.

    Values: zeroshot, fft, qv, attention or full.
    """
    df = df.copy()

    strategy = df["adaptation"].map({"none": "zeroshot", "fft": "fft"})
    strategy = strategy.fillna(df["lora_target"])

    df["strategy"] = strategy

    return df


def is_collapsed(df):
    """Return a boolean Series marking collapsed runs (index as `df`).

    A run is collapsed if its metric is below its dataset's zero-shot
    mean, or below 50% of the median of the healthy sibling seeds in
    its cell (CELL_COLS).

    Since "healthy" depends on the result, the rule is applied
    iteratively: start with the runs below the zero-shot mean, then
    recompute the median over the unflagged seeds until the flagged
    set no longer changes. Zero-shot runs are never flagged.
    """
    required = set(CELL_COLS) | {"metric", "dataset", "adaptation"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"is_collapsed() needs columns {sorted(missing)}")

    zeroshot_mean = (
        df.loc[df["adaptation"] == "none"]
        .groupby("dataset")["metric"]
        .mean()
    )

    result = pd.Series(False, index=df.index)

    trainable = df[df["adaptation"] != "none"]

    for _, group in trainable.groupby(CELL_COLS, dropna=False):

        dataset = group["dataset"].iloc[0]
        zeroshot_threshold = zeroshot_mean.get(dataset, float("-inf"))

        below_zeroshot = group["metric"] < zeroshot_threshold

        flagged = below_zeroshot.copy()

        for _ in range(len(group)):

            healthy_metric = group.loc[~flagged, "metric"]

            if healthy_metric.empty:
                break

            sibling_median = healthy_metric.median()
            below_sibling = group["metric"] < 0.5 * sibling_median

            new_flagged = below_zeroshot | below_sibling

            if new_flagged.equals(flagged):
                break

            flagged = new_flagged

        result.loc[group.index] = flagged

    return result


def aggregate_cells(df):
    """Summarize each cell: mean, std, n_seeds and n_collapsed."""
    collapsed = is_collapsed(df)
    working = df.assign(_collapsed=collapsed)

    grouped = working.groupby(CELL_COLS, dropna=False)

    summary = grouped["metric"].agg(
        mean_metric="mean",
        std_metric="std",
        n_seeds="count",
    ).reset_index()

    n_collapsed = (
        grouped["_collapsed"].sum()
        .reset_index(name="n_collapsed")
    )

    return summary.merge(n_collapsed, on=CELL_COLS)
