"""Plot a quick overview of the small-model pilot runs.

Reads results/other_models_runs.csv (see aggregate_other_models.py).
Not a thesis figure: a single-seed pilot on four other base models, so
it is not in plots.py's FIGURES or the figure inventory. The label and
style helpers of the thesis figures are reused.

Usage:
    python -m scripts.plot_other_models_overview
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.analysis.data import attach_strategy
from src.analysis.plotstyle import METRIC_NAMES, CATEGORICAL_PALETTE, apply_style, categorical_color, pretty


DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
DATASET_LABELS = {"mnli": "MNLI", "squad": "SQuAD", "conll2003": "CoNLL-2003", "gsm8k": "GSM8K"}
STRATEGIES = ["zeroshot", "fft", "qv", "attention", "full"]

IN_CSV = "results/other_models_runs.csv"
OUT_PNG = "results/other_models_overview.png"


def main():

    apply_style()

    df = pd.read_csv(IN_CSV)
    df = attach_strategy(df)

    models = sorted(df["model"].unique())
    model_colors = categorical_color(models)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    width = 0.8 / len(models)

    for ax, dataset in zip(axes.flat, DATASETS):

        sub = df[df["dataset"] == dataset]

        for m_idx, model in enumerate(models):

            offsets = [s_idx + (m_idx - (len(models) - 1) / 2) * width for s_idx in range(len(STRATEGIES))]
            values = [
                sub.loc[(sub["model"] == model) & (sub["strategy"] == s), "metric"]
                for s in STRATEGIES
            ]
            # Single seed: one run per cell, or a gap (not a zero bar)
            # if that run never finished.
            values = [v.iloc[0] if len(v) else float("nan") for v in values]

            ax.bar(offsets, values, width=width, color=model_colors[model], label=model, edgecolor="black", linewidth=0.3)

        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels([pretty(s) for s in STRATEGIES], rotation=30, ha="right")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylabel(METRIC_NAMES[dataset])
        ax.set_ylim(0, 1)

    # "outside" lets constrained_layout (enabled by apply_style())
    # reserve room for the legend; tight_layout() would conflict.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(models), frameon=False)

    fig.savefig(OUT_PNG, dpi=200)

    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
