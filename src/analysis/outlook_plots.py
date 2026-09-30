"""Overview figure of the cross-model outlook runs.

Shows the three models that completed their grid cleanly
(llama-3.1-8b-instruct, deepseek-r1-distill-qwen-7b and
mistral-7b-instruct-v0.3). Kept separate from plots.py because the
outlook data has its own schema (a `model` column, one seed, a 3-point
grid). It uses the shared plotstyle helpers and is saved next to the
thesis figures, but is not registered in FIGURES.

Usage:
    python -m src.analysis.outlook_plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.data import attach_strategy
from src.analysis.plotstyle import (
    METRIC_NAMES,
    STRATEGY_COLORS,
    TEXTWIDTH_IN,
    apply_style,
    pretty,
    save_figure,
)

OUTLOOK_RESULTS_CSV = "results/outlook_runs.csv"

# Models that completed all 44 runs cleanly. Left out, but still in
# the results: gpt-oss-20b (FFT out of memory; "full" equals
# "attention") and gemma-4-e4b-it (not supported by PEFT; FFT kernel
# error).
OUTLOOK_MODELS = [
    "llama-3.1-8b-instruct",
    "deepseek-r1-distill-qwen-7b",
    "mistral-7b-instruct-v0.3",
]

# Zero-shot is drawn as a bar here instead of a reference line, since
# every model has its own zero-shot baseline.
OUTLOOK_STRATEGIES = ["zeroshot", "fft", "qv", "attention", "full"]

MAIN_DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
MAIN_DATASET_LABELS = {
    "mnli": "MNLI", "squad": "SQuAD",
    "conll2003": "CoNLL-2003", "gsm8k": "GSM8K",
}


def _load_outlook_runs(path=OUTLOOK_RESULTS_CSV):
    df = pd.read_csv(path)

    missing_models = set(OUTLOOK_MODELS) - set(df["model"].unique())

    assert not missing_models, (
        f"{path} is missing expected outlook model(s) {sorted(missing_models)} -- "
        f"regenerate with scripts/aggregate_other_models.py --results-root "
        f"results/outlook --out {path}"
    )

    return df


def _target_collapsed_to_attention(df):
    """Return the models whose "full" target equals "attention".

    This happens when the MLP has no up/down/gate_proj modules (e.g.
    the MoE layers of gpt-oss-20b), so both targets train the same
    parameters. Detected from trainable_params, not hard-coded.
    """
    lora = df[df["adaptation"] == "lora"]
    flagged = set()

    for model, sub in lora.groupby("model"):

        attention_params = sub[sub["lora_target"] == "attention"].set_index("rank")["trainable_params"]
        full_params = sub[sub["lora_target"] == "full"].set_index("rank")["trainable_params"]

        shared_ranks = attention_params.index.intersection(full_params.index)

        if len(shared_ranks) > 0 and (attention_params.loc[shared_ranks] == full_params.loc[shared_ranks]).all():
            flagged.add(model)

    return flagged


def _outlook_summary(df):
    """Return the best metric per (model, dataset, strategy).

    Zero-shot and FFT have one run each; the LoRA targets take the best
    of the three (r, alpha) points. Single seed, so no error bars.
    """
    df = attach_strategy(df)

    return (
        df.groupby(["model", "dataset", "strategy"])["metric"]
        .max()
        .reset_index()
    )


def fig_outlook_overview(df):
    """Plot the best metric per model and strategy as grouped bars.

    One panel per dataset, one group of five strategy bars per model in
    OUTLOOK_MODELS. The hatching for a "full" target that equals
    "attention" is kept for models added later.
    """
    summary = _outlook_summary(df)
    hatched_models = _target_collapsed_to_attention(df)

    n_models = len(OUTLOOK_MODELS)
    n_strategies = len(OUTLOOK_STRATEGIES)
    bar_width = 0.8 / n_strategies
    group_centers = np.arange(n_models)

    fig, axes = plt.subplots(
        1, len(MAIN_DATASETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.5),
    )

    for ax, dataset in zip(axes, MAIN_DATASETS):

        cell = summary[summary["dataset"] == dataset]

        for k, strategy in enumerate(OUTLOOK_STRATEGIES):

            offset = (k - (n_strategies - 1) / 2) * bar_width
            values, hatches = [], []

            for model in OUTLOOK_MODELS:

                match = cell[(cell["model"] == model) & (cell["strategy"] == strategy)]

                if match.empty:
                    values.append(np.nan)
                    hatches.append(None)
                    continue

                values.append(match["metric"].iloc[0])
                is_hatched = strategy == "full" and model in hatched_models
                hatches.append("///" if is_hatched else None)

            for x, value, hatch in zip(group_centers + offset, values, hatches):

                if np.isnan(value):
                    continue

                ax.bar(
                    x, value, width=bar_width,
                    color=STRATEGY_COLORS[strategy], hatch=hatch,
                    edgecolor="white", linewidth=0.4,
                )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [pretty(m) for m in OUTLOOK_MODELS], rotation=30, ha="right", fontsize=6,
        )
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(METRIC_NAMES[dataset])
        ax.set_ylim(0, 1.0)

    # Build the legend from OUTLOOK_STRATEGIES, so a strategy missing
    # for the first model still gets an entry.
    strategy_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STRATEGY_COLORS[s])
        for s in OUTLOOK_STRATEGIES
    ]
    strategy_labels = [pretty(s) for s in OUTLOOK_STRATEGIES]

    # No hatch legend entry, since no current model is hatched. Add one
    # if a hatched model is added to OUTLOOK_MODELS.
    fig.legend(
        strategy_handles, strategy_labels, loc="outside upper center",
        ncol=len(OUTLOOK_STRATEGIES), frameon=False, fontsize=6,
    )

    return fig, "outlook_models_overview"


def main():

    apply_style()

    df = _load_outlook_runs()

    fig, slug = fig_outlook_overview(df)

    pdf_path, png_path = save_figure(fig, "fig_outlook_overview", slug)

    plt.close(fig)

    print(f"fig_outlook_overview: wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
