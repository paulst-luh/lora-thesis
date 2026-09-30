"""Entry point for all thesis figures.

Each figure is a function `fig_<id>(df) -> (Figure, slug)` registered
in FIGURES; adding a figure means writing the function and adding one
line there.

Usage:
    python -m src.analysis.plots --figure fig_smoke
    python -m src.analysis.plots --all
"""

import argparse
import functools
import glob
import json
import os

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

from src.analysis.data import CELL_COLS, attach_strategy, is_collapsed, load_runs
from src.analysis.transfer import (
    best_config,
    grid_median_regret,
    leave_one_out_default,
    percentile_rank,
    regret,
    winner_seed_sd,
)
from src.analysis.raw_metrics import SUB_METRIC_KEYS, load_sub_metrics
from src.analysis.shapley import (
    LADDER_CONTRIBUTORS,
    MAIN_CONTRIBUTORS,
    RANK_RATIO_CONTRIBUTORS,
    all_ladder_attributions,
    all_main_sweep_attributions,
    all_rank_ratio_attributions,
    ladder_attribution,
    main_sweep_attribution,
)
from src.analysis.plotstyle import (
    CATEGORICAL_PALETTE,
    MAX_FIG_HEIGHT_IN,
    METRIC_NAMES,
    NUMERICAL_ZERO,
    RANK_VALUES,
    STRATEGY_COLORS,
    TEXTWIDTH_IN,
    adjust_lightness,
    annot_color,
    apply_style,
    categorical_axis,
    categorical_color,
    fmt_e3,
    fmt_lr,
    fmt_subset,
    pretty,
    rank_color,
    save_figure,
)


# RQ1 ladder study. The ladder blocks contain exactly this design, so
# filtering on `block` is enough.

LADDER_BLOCKS = ["ladder_isoepoch", "ladder_isostep"]
# The full-target ladder: same design with lora_target "full". Kept
# separate so LADDER_BLOCKS stays attention-only; only fig_6_3e pools
# both.
LADDER_FULL_BLOCKS = ["ladder_isoepoch_full", "ladder_isostep_full"]
LADDER_DATASETS = ["mnli", "squad"]
LADDER_SUBSET_SIZES = [1000, 5000, 20000, 50000]
LADDER_LEARNING_RATES = [1e-4, 3e-4]
TRAINING_REGIME_ORDER = ["iso_epoch", "iso_step"]
TRAINING_REGIME_COLORS = categorical_color(TRAINING_REGIME_ORDER)
DATASET_LABELS = {"mnli": "MNLI", "squad": "SQuAD"}

# Colors of the two ladder learning rates, taken from the categorical
# palette so they never suggest a strategy.
LADDER_LR_COLORS = categorical_color(LADDER_LEARNING_RATES)

# Marker per seed for fig_6_3d; the core ladder uses seeds 42-44 only.
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}


def _ladder_data(df):
    return df[df["block"].isin(LADDER_BLOCKS)]


def _ladder_data_with_full(df):
    """Return both ladder targets (attention and full) pooled.

    lora_target tells them apart. Used only by fig_6_3e.
    """
    return df[df["block"].isin(LADDER_BLOCKS + LADDER_FULL_BLOCKS)]


# RQ3 main sweep. These blocks contain exactly 4 datasets x 3 targets
# x 10 (rank, alpha) pairs x 3 learning rates x 3 seeds, so filtering
# on `block` is enough. Every rank has both scaling ratios 1 and 2.

MAIN_SWEEP_BLOCKS = ["main_qv", "main_attention", "main_full"]
MAIN_DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
MAIN_TARGETS = ["qv", "attention", "full"]
MAIN_RANK_ALPHA_PAIRS = [
    (4, 4), (4, 8), (8, 8), (8, 16), (16, 16),
    (16, 32), (32, 32), (32, 64), (64, 64), (64, 128),
]
MAIN_LEARNING_RATES = [1e-4, 3e-4, 5e-4]
# Configurations per (dataset, target) in the main sweep.
MAIN_N_CONFIGS = len(MAIN_RANK_ALPHA_PAIRS) * len(MAIN_LEARNING_RATES)

MAIN_DATASET_LABELS = {
    "mnli": "MNLI", "squad": "SQuAD",
    "conll2003": "CoNLL-2003", "gsm8k": "GSM8K",
}


def _main_sweep_data(df):
    return df[df["block"].isin(MAIN_SWEEP_BLOCKS)]


def _main_sweep_data_with_risky(df):
    """Return the main sweep plus the risky-seed reruns (n=10 cells).

    Used only by fig_6_5b; all other figures keep the uniform 3-seed
    main sweep of _main_sweep_data().
    """
    return df[df["block"].isin(MAIN_SWEEP_BLOCKS + RISKY_SEED_BLOCKS)]


def _main_sweep_collapse(df):
    """Return is_collapsed() for the main-sweep rows.

    Computed on a frame that still contains the zero-shot rows, since
    is_collapsed() takes its zero-shot threshold from them.
    """
    eligible = df[df["block"].isin(MAIN_SWEEP_BLOCKS + ["zeroshot"])]
    collapsed = is_collapsed(eligible)

    return collapsed.loc[_main_sweep_data(df).index]


# Risky-seed block -> its parent main-sweep block. Relabeling lets the
# extra seeds share one cell (and one sibling median) with the original
# three; used only by _main_sweep_collapse_with_risky().
_RISKY_BLOCK_PARENT = {
    "main_attention_riskyseeds": "main_attention",
    "main_attention_riskyseeds_lr3e4": "main_attention",
    "main_attention_riskyseeds_lr5e4": "main_attention",
    "main_full_riskyseeds": "main_full",
    "main_full_riskyseeds_lr3e4": "main_full",
    "main_full_riskyseeds_lr5e4": "main_full",
}


def _main_sweep_collapse_with_risky(df):
    """Like _main_sweep_collapse(), with the risky seeds pooled in.

    Risky-seed blocks are relabeled to their parent block for the
    sibling-seed grouping (see _RISKY_BLOCK_PARENT).
    """
    eligible = df[df["block"].isin(MAIN_SWEEP_BLOCKS + RISKY_SEED_BLOCKS + ["zeroshot"])].copy()
    eligible["block"] = eligible["block"].replace(_RISKY_BLOCK_PARENT)
    collapsed = is_collapsed(eligible)

    return collapsed.loc[_main_sweep_data_with_risky(df).index]


# Chapter 6.1 strategy comparison. CORE_BLOCKS is the clean 5-block
# design without diagnostic rounds. FFT_STABILITY_BLOCKS pools
# fft_wd0.1 and its extra seeds into an n=10 sample at one weight decay
# (fig_6_1b/6_1c).

CORE_BLOCKS = ["zeroshot", "fft", "main_qv", "main_attention", "main_full"]
STRATEGY_ORDER = ["zeroshot", "fft", "qv", "attention", "full"]
FFT_LEARNING_RATES = [1e-5, 3e-5, 5e-5]
FFT_STABILITY_BLOCKS = ["fft_wd0.1", "fft_wd0.1_moreseeds"]

# Risky-seed reruns: seeds 45-51 added to the collapse-prone
# attention/full cells of the main sweep, giving n=10.
RISKY_SEED_BLOCKS = [
    "main_attention_riskyseeds", "main_attention_riskyseeds_lr3e4", "main_attention_riskyseeds_lr5e4",
    "main_full_riskyseeds", "main_full_riskyseeds_lr3e4", "main_full_riskyseeds_lr5e4",
]


def _core_strategy_data(df):
    core = df[df["block"].isin(CORE_BLOCKS)]
    return attach_strategy(core)


def _zeroshot_means(df):
    """Return the zero-shot mean metric per dataset.

    Shared by every figure that draws a zero-shot reference line.
    """
    return df[df["block"] == "zeroshot"].groupby("dataset")["metric"].mean()


def _draw_zeroshot_line(ax, zeroshot_mean, label=None):
    """Draw the dashed zero-shot reference line on `ax`.

    Pass `label` only on the panel the legend is built from.
    """
    ax.axhline(
        zeroshot_mean, color=STRATEGY_COLORS["zeroshot"],
        linestyle="--", linewidth=1.2, alpha=0.7, zorder=0, label=label,
    )


def _fft_stability_data(df):
    """Return fft_wd0.1 plus its extra seeds as one n=10 block.

    The shared `block` label makes is_collapsed() treat them as one
    cell.
    """
    stability = df[df["block"].isin(FFT_STABILITY_BLOCKS)].copy()
    stability["block"] = "fft_stability"

    return stability


def _wilson_ci(k, n, z=1.959963984540054):
    """Return the Wilson score interval for k/n (95% by default).

    Unlike the normal approximation, it stays within [0, 1], which
    matters for the many 0/n cells.
    """
    if n == 0:
        return (float("nan"), float("nan"))

    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * ((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) ** 0.5)

    return (max(0.0, center - margin), min(1.0, center + margin))


def fig_smoke(df):
    """Smoke test, not a thesis figure: parameters vs. metric.

    Checks that the style, save and CLI path works. Zero-shot is left
    out because its zero trainable parameters cannot go on a log axis.
    """
    working = attach_strategy(df)
    working = working[working["adaptation"] != "none"]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.6))

    for strategy, color in STRATEGY_COLORS.items():

        subset = working[working["strategy"] == strategy]

        if subset.empty:
            continue

        ax.scatter(
            subset["trainable_params"],
            subset["metric"],
            color=color,
            label=pretty(strategy),
            s=8,
            alpha=0.5,
            linewidths=0,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)

    return fig, "smoke_trainable_params_vs_metric"


def fig_5_1(df):
    """Plot realized optimizer steps vs. subset size per regime.

    Facets MNLI and SQuAD (ladder blocks). The step count is fixed by
    dataset, regime and subset size, so each point is a constant. The
    figure shows iso_step flat at 938 steps and iso_epoch growing from
    189 to 9375, which separates more data from more steps.
    """
    ladder = _ladder_data(df)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.55),
        sharey=True,
    )

    regime_markers = {"iso_epoch": "o", "iso_step": "s"}

    for ax, dataset in zip(axes, LADDER_DATASETS):

        subset = ladder[ladder["dataset"] == dataset]

        for regime in TRAINING_REGIME_ORDER:

            steps = (
                subset[subset["training_regime"] == regime]
                .groupby("subset_size")["optimizer_steps"]
                .mean()
                .sort_index()
            )

            ax.plot(
                steps.index, steps.values,
                color=TRAINING_REGIME_COLORS[regime], marker=regime_markers[regime],
                markersize=4, label=pretty(regime),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(LADDER_SUBSET_SIZES)
        ax.set_xticklabels([fmt_subset(s) for s in LADDER_SUBSET_SIZES], rotation=45)
        ax.set_xlabel(pretty("subset_size"))
        ax.set_title(DATASET_LABELS[dataset])

    axes[0].set_ylabel("Optimizer steps")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=2,
        frameon=False,
    )

    return fig, "schedule_verification"


def _fig_6_3a_core(df, lr_filter):
    """Shared implementation of fig_6_3a and fig_6_3a_lr3e4.

    The zero-shot mean is far below all rank curves, so it is shown as
    a number in each panel's corner instead of a reference line.
    """
    ladder = _ladder_data(df)
    ladder = ladder[ladder["learning_rate"] == lr_filter]
    zeroshot_means = _zeroshot_means(df)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
        sharex=True, sharey="row",
    )

    for row_idx, dataset in enumerate(LADDER_DATASETS):
        for col_idx, regime in enumerate(TRAINING_REGIME_ORDER):

            ax = axes[row_idx, col_idx]

            cell = ladder[
                (ladder["dataset"] == dataset)
                & (ladder["training_regime"] == regime)
            ]

            for rank in RANK_VALUES:

                rank_cell = cell[cell["rank"] == rank]

                stats = (
                    rank_cell.groupby("subset_size")["metric"]
                    .agg(["mean", "std"])
                    .sort_index()
                )

                color = rank_color(rank)

                ax.plot(
                    stats.index, stats["mean"],
                    color=color, marker="o", markersize=3,
                    label=f"r={rank}",
                )
                ax.fill_between(
                    stats.index,
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    color=color, alpha=0.15, linewidth=0,
                )

            ax.text(
                0.03, 0.03, f"Zero-shot mean: {zeroshot_means[dataset]:.3f}",
                ha="left", va="bottom", fontsize=5.5, transform=ax.transAxes,
            )

            ax.set_xscale("log")
            ax.set_xticks(LADDER_SUBSET_SIZES)
            ax.set_xticklabels([fmt_subset(s) for s in LADDER_SUBSET_SIZES], rotation=45)

            if row_idx == 0:
                ax.set_title(pretty(regime))

            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\n{METRIC_NAMES[dataset]}")

            if row_idx == 1:
                ax.set_xlabel(pretty("subset_size"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=len(RANK_VALUES),
        frameon=False,
    )

    return fig


def fig_6_3a(df):
    """Plot metric vs. training-set size per rank at lr 1e-4.

    Main figure of the chapter: rows MNLI/SQuAD, columns iso_epoch/
    iso_step, one line per rank with a +-1 SD band over seeds, y-axes
    shared per row.

    At this learning rate rank 64 does not degrade with subset size on
    MNLI. The degradation only appears at lr 3e-4 (fig_6_3a_lr3e4) and
    comes from single-seed collapses (fig_6_3d): an LR x rank
    interaction, not a pure rank effect.
    """
    fig = _fig_6_3a_core(df, lr_filter=1e-4)

    return fig, "ladder_metric_vs_subset_size"


def fig_6_3a_lr3e4(df):
    """Plot fig_6_3a at lr 3e-4 (appendix variant).

    Here rank 64 becomes unstable through single-seed collapses, mostly
    at larger subset sizes (see fig_6_3d).
    """
    fig = _fig_6_3a_core(df, lr_filter=3e-4)

    return fig, "ladder_metric_vs_subset_size_lr3e4"


def fig_6_3c(df):
    """Plot a heatmap of the best rank per ladder rung.

    Rows are (dataset, regime, learning rate), columns subset sizes.
    The text is the winning rank, the color its gap to the runner-up.
    Hatched cells have a gap within the winner's seed SD, i.e. not
    significant. Annotations get an outline so they stay legible on the
    hatching.
    """
    ladder = _ladder_data(df)

    per_rank = (
        ladder.groupby(
            ["dataset", "training_regime", "learning_rate", "subset_size", "rank"]
        )["metric"]
        .agg(["mean", "std"])
        .reset_index()
    )

    row_keys = [
        (dataset, regime, lr)
        for dataset in LADDER_DATASETS
        for regime in TRAINING_REGIME_ORDER
        for lr in LADDER_LEARNING_RATES
    ]

    n_rows, n_cols = len(row_keys), len(LADDER_SUBSET_SIZES)

    gap_matrix = np.full((n_rows, n_cols), np.nan)
    rank_matrix = np.full((n_rows, n_cols), np.nan)
    reliable_matrix = np.zeros((n_rows, n_cols), dtype=bool)

    for i, (dataset, regime, lr) in enumerate(row_keys):
        for j, subset_size in enumerate(LADDER_SUBSET_SIZES):

            cell = per_rank[
                (per_rank["dataset"] == dataset)
                & (per_rank["training_regime"] == regime)
                & (per_rank["learning_rate"] == lr)
                & (per_rank["subset_size"] == subset_size)
            ].sort_values("mean", ascending=False)

            best = cell.iloc[0]
            second_mean = cell.iloc[1]["mean"] if len(cell) > 1 else np.nan
            gap = best["mean"] - second_mean
            best_std = best["std"] if pd.notna(best["std"]) else 0.0

            gap_matrix[i, j] = gap
            rank_matrix[i, j] = best["rank"]
            reliable_matrix[i, j] = gap >= best_std

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.75))

    # cividis encodes the gap, not a rank, so it differs on purpose from
    # the viridis rank colors.
    im = ax.imshow(gap_matrix, cmap="cividis", aspect="auto")
    fig.colorbar(im, ax=ax, label=r"Gap to 2nd-best rank ($\Delta$ score)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([fmt_subset(s) for s in LADDER_SUBSET_SIZES])
    ax.set_xlabel(pretty("subset_size"))

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([
        f"{DATASET_LABELS[dataset]} | {pretty(regime)} | {fmt_lr(lr)}"
        for dataset, regime, lr in row_keys
    ])

    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):

            text_color = annot_color(gap_matrix[i, j], im.cmap, norm)
            stroke_color = "black" if text_color == "white" else "white"

            ax.text(
                j, i, f"{int(rank_matrix[i, j])}", ha="center", va="center",
                color=text_color, fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground=stroke_color)],
            )

            if not reliable_matrix[i, j]:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="////", edgecolor=text_color, linewidth=0,
                ))

    legend_handles = [
        Patch(facecolor="none", edgecolor="black", hatch="////", label="Winning margin $<$ seed SD (not significant)"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=1,
        frameon=False,
    )

    return fig, "ladder_best_rank_per_rung"


def fig_6_3d(df):
    """Plot every rank-64 seed across subset sizes (no means).

    Rows are datasets, columns regimes; color is the learning rate and
    marker the seed. Shows whether a dip (e.g. SQuAD at iso_epoch) comes
    from one collapsing seed or from all seeds.
    """
    ladder = _ladder_data(df)
    r64 = ladder[ladder["rank"] == 64]
    zeroshot_means = _zeroshot_means(df)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
        sharex=True, sharey="row",
    )

    for row_idx, dataset in enumerate(LADDER_DATASETS):
        for col_idx, regime in enumerate(TRAINING_REGIME_ORDER):

            ax = axes[row_idx, col_idx]

            _draw_zeroshot_line(ax, zeroshot_means[dataset])

            cell = r64[
                (r64["dataset"] == dataset)
                & (r64["training_regime"] == regime)
            ]

            for lr, color in LADDER_LR_COLORS.items():

                lr_cell = cell[cell["learning_rate"] == lr]

                for seed, marker in SEED_MARKERS.items():

                    seed_cell = lr_cell[lr_cell["seed"] == seed]

                    ax.scatter(
                        seed_cell["subset_size"], seed_cell["metric"],
                        color=color, marker=marker,
                        s=18, alpha=0.8, linewidths=0,
                    )

            ax.set_xscale("log")
            ax.set_xticks(LADDER_SUBSET_SIZES)
            ax.set_xticklabels([fmt_subset(s) for s in LADDER_SUBSET_SIZES], rotation=45)

            if row_idx == 0:
                ax.set_title(pretty(regime))

            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\n{METRIC_NAMES[dataset]}")

            if row_idx == 1:
                ax.set_xlabel(pretty("subset_size"))

    # Two legends, since each point encodes both the learning rate
    # (color) and the seed (marker).
    lr_handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="", markersize=5, label=fmt_lr(lr))
        for lr, color in LADDER_LR_COLORS.items()
    ]
    seed_handles = [
        Line2D([0], [0], marker=marker, color="0.3", linestyle="", markersize=5, label=f"seed {seed}")
        for seed, marker in SEED_MARKERS.items()
    ]
    zeroshot_handle = [
        Line2D([0], [0], color=STRATEGY_COLORS["zeroshot"], linestyle="--", linewidth=1.2, label="Zero-shot mean"),
    ]
    # Tighter spacing and a smaller font keep this 7-entry legend within
    # the text width (the figure is saved without cropping).
    fig.legend(
        handles=lr_handles + seed_handles + zeroshot_handle, loc="outside upper center",
        ncol=len(lr_handles) + len(seed_handles) + 1, frameon=False,
        fontsize=6.5, handletextpad=0.4, columnspacing=1.0, handlelength=1.5,
    )

    return fig, "ladder_r64_seed_level"


def fig_6_3e(df):
    """Overlay the full-target ladder on the attention ladder.

    Same layout as fig_6_3a at lr 1e-4; line color is the rank, line
    style the target (solid attention, dashed full). Tests whether full
    degrades at the same subset sizes and regimes as attention. The
    zero-shot mean is shown as corner text, as in fig_6_3a.
    """
    ladder = _ladder_data_with_full(df)
    ladder = ladder[ladder["learning_rate"] == 1e-4]
    zeroshot_means = _zeroshot_means(df)

    targets = ["attention", "full"]
    target_linestyles = {"attention": "-", "full": "--"}

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
        sharex=True, sharey="row",
    )

    for row_idx, dataset in enumerate(LADDER_DATASETS):
        for col_idx, regime in enumerate(TRAINING_REGIME_ORDER):

            ax = axes[row_idx, col_idx]

            cell = ladder[
                (ladder["dataset"] == dataset)
                & (ladder["training_regime"] == regime)
            ]

            for target in targets:

                target_cell = cell[cell["lora_target"] == target]
                linestyle = target_linestyles[target]

                for rank in RANK_VALUES:

                    rank_cell = target_cell[target_cell["rank"] == rank]

                    if rank_cell.empty:
                        continue

                    stats = (
                        rank_cell.groupby("subset_size")["metric"]
                        .agg(["mean", "std"])
                        .sort_index()
                    )

                    color = rank_color(rank)

                    ax.plot(
                        stats.index, stats["mean"],
                        color=color, linestyle=linestyle, marker="o", markersize=3,
                    )
                    ax.fill_between(
                        stats.index,
                        stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"],
                        color=color, alpha=0.12, linewidth=0,
                    )

            ax.text(
                0.03, 0.03, f"Zero-shot mean: {zeroshot_means[dataset]:.3f}",
                ha="left", va="bottom", fontsize=5.5, transform=ax.transAxes,
            )

            ax.set_xscale("log")
            ax.set_xticks(LADDER_SUBSET_SIZES)
            ax.set_xticklabels([fmt_subset(s) for s in LADDER_SUBSET_SIZES], rotation=45)

            # State the learning rate in the panel (no figure titles),
            # since the PNG may be used without its caption.
            if row_idx == 0 and col_idx == 0:
                ax.text(
                    0.03, 0.97, f"Learning rate {fmt_lr(1e-4)}",
                    ha="left", va="top", fontsize=6, transform=ax.transAxes,
                )

            if row_idx == 0:
                ax.set_title(pretty(regime))

            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\n{METRIC_NAMES[dataset]}")

            if row_idx == 1:
                ax.set_xlabel(pretty("subset_size"))

    rank_handles = [
        Line2D([0], [0], color=rank_color(rank), linestyle="-", marker="o", markersize=3, label=f"r={rank}")
        for rank in RANK_VALUES
    ]
    target_handles = [
        Line2D([0], [0], color="black", linestyle=target_linestyles[t], label=pretty(t))
        for t in targets
    ]
    # Tighter spacing and a smaller font keep this 7-entry legend within
    # the text width (the figure is saved without cropping).
    fig.legend(
        handles=rank_handles + target_handles, loc="outside upper center",
        ncol=len(RANK_VALUES) + len(targets), frameon=False,
        fontsize=6.5, handletextpad=0.4, columnspacing=1.0, handlelength=1.5,
    )

    return fig, "ladder_full_vs_attention"


def _fig_6_5a_single(df, dataset):
    """Plot metric vs. learning rate per (rank, alpha) for one dataset.

    One panel per target, y-axes shared, seed points overlaid. Color is
    the rank; scaling ratio 1 is dashed and lighter than ratio 2. Lines
    that cross show that the rank order depends on the learning rate.

    Each panel has an inset zoomed to the non-collapsed points (via
    is_collapsed()), placed in an empty strip right of the data.
    """
    main = _main_sweep_data(df)
    collapsed = _main_sweep_collapse(df)
    main = main.assign(_collapsed=collapsed)
    zeroshot_mean = _zeroshot_means(df)[dataset]

    fig, axes = plt.subplots(
        1, len(MAIN_TARGETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.60),
        sharex=True, sharey=True,
    )

    for col_idx, target in enumerate(MAIN_TARGETS):

        ax = axes[col_idx]

        # Evenly spaced categories instead of a log axis, on which 3e-4
        # and 5e-4 would crowd together.
        lr_positions = categorical_axis(ax, MAIN_LEARNING_RATES, fmt_lr)

        # Extend xlim past the last category, leaving an empty strip on
        # the right for the inset that no data line can enter.
        n_lr = len(MAIN_LEARNING_RATES)
        ax.set_xlim(-0.5, n_lr - 0.5 + 1.7)

        # Rotated, since the widened xlim leaves less room per tick.
        ax.set_xticklabels([fmt_lr(lr) for lr in MAIN_LEARNING_RATES], rotation=45, ha="right")

        # Main panel only: the zero-shot line would undo the inset's
        # zoom on the healthy band.
        _draw_zeroshot_line(
            ax, zeroshot_mean,
            label="Zero-shot mean" if col_idx == 0 else None,
        )

        cell = main[
            (main["dataset"] == dataset) & (main["lora_target"] == target)
        ]

        # Tall and narrow, in the empty right margin created above.
        inset = ax.inset_axes([0.60, 0.08, 0.37, 0.86])
        inset.set_xticks(list(lr_positions.values()))
        inset.set_xticklabels([])
        inset.tick_params(labelsize=6.5, pad=1.5)
        inset.yaxis.set_major_locator(plt.MaxNLocator(3))
        # The inset's own x-range, not the widened one of the panel.
        inset.set_xlim(-0.5, n_lr - 0.5)

        healthy_cell = cell[~cell["_collapsed"]]

        for pair_idx, (rank, alpha) in enumerate(MAIN_RANK_ALPHA_PAIRS):

            pair_cell = cell[(cell["rank"] == rank) & (cell["alpha"] == alpha)]

            if pair_cell.empty:
                continue

            ratio = alpha / rank
            linestyle = "-" if ratio == 2 else "--"
            base_color = rank_color(rank)
            # Ratio 1 (dashed) is lighter than ratio 2 (solid).
            color = base_color if ratio == 2 else adjust_lightness(base_color, 1.35)

            # Small fixed jitter per (rank, alpha) pair against overlap.
            jitter = 0.015 * (pair_idx - (len(MAIN_RANK_ALPHA_PAIRS) - 1) / 2)

            means = pair_cell.groupby("learning_rate")["metric"].mean().sort_index()
            mean_x = [lr_positions[lr] + jitter for lr in means.index]

            ax.plot(
                mean_x, means.values,
                color=color, linestyle=linestyle, marker="o", markersize=3,
                label=f"r={rank:g}, ratio={ratio:g}",
            )

            point_x = pair_cell["learning_rate"].map(lr_positions) + jitter

            ax.scatter(
                point_x, pair_cell["metric"],
                color=color, s=6, alpha=0.35, linewidths=0,
            )

            healthy_pair_cell = healthy_cell[
                (healthy_cell["rank"] == rank) & (healthy_cell["alpha"] == alpha)
            ]

            if healthy_pair_cell.empty:
                continue

            healthy_means = healthy_pair_cell.groupby("learning_rate")["metric"].mean().sort_index()
            healthy_x = [lr_positions[lr] + jitter for lr in healthy_means.index]

            inset.plot(
                healthy_x, healthy_means.values,
                color=color, linestyle=linestyle, marker="o", markersize=2,
                linewidth=1.0,
            )

        if not healthy_cell.empty:
            lo, hi = healthy_cell["metric"].min(), healthy_cell["metric"].max()
            pad = (hi - lo) * 0.1 or 0.01
            inset.set_ylim(lo - pad, hi + pad)

        ax.set_title(pretty(target))

        if col_idx == 0:
            ax.set_ylabel(METRIC_NAMES[dataset])

        ax.set_xlabel(pretty("learning_rate"))

    # Tighter spacing and a smaller font keep the 11 legend entries
    # within the text width.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=5,
        frameon=False, fontsize=6.5, handletextpad=0.4,
        columnspacing=1.0, handlelength=1.5,
    )

    return fig, f"lr_interaction_rank_target_{dataset}"


def fig_6_5b(df):
    """Plot the collapse rate per cell of the main sweep.

    One panel per target; rows are the (rank, alpha) pairs, columns the
    learning rates grouped by dataset. The color is the fraction of
    collapsed seeds (is_collapsed()), which shows the discrete failure
    mode better than a mean or SD would.

    Cells without collapse stay blank. The text is the number of
    collapsed seeds out of 3; cells with n=10 (risky-seed reruns) get a
    heavier border. OrRd is fixed to [0, 1], so a clean panel always
    looks clean.
    """
    main = _main_sweep_data_with_risky(df)
    collapsed = _main_sweep_collapse_with_risky(df)

    working = main.assign(_collapsed=collapsed)

    per_cell = (
        working.groupby(
            ["dataset", "lora_target", "rank", "alpha", "learning_rate"]
        )["_collapsed"]
        .agg(n_collapsed="sum", n="count")
        .reset_index()
    )
    per_cell["fraction"] = per_cell["n_collapsed"] / per_cell["n"]

    # Layout: dataset groups of n_lr columns, separated by n_gap_cols
    # blank columns so the group headers do not overlap.
    n_rows = len(MAIN_RANK_ALPHA_PAIRS)
    n_lr = len(MAIN_LEARNING_RATES)
    n_datasets = len(MAIN_DATASETS)
    n_gap_cols = 2
    group_stride = n_lr + n_gap_cols
    n_cols = n_datasets * group_stride - n_gap_cols  # No gap after the last group.

    cmap = plt.get_cmap("OrRd")
    norm = plt.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(
        1, len(MAIN_TARGETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.62),
        sharey=True,
    )

    for col_idx, target in enumerate(MAIN_TARGETS):

        ax = axes[col_idx]

        matrix = np.full((n_rows, n_cols), np.nan)
        counts = np.zeros((n_rows, n_cols), dtype=int)
        n_seeds = np.zeros((n_rows, n_cols), dtype=int)

        for d, dataset in enumerate(MAIN_DATASETS):
            for i, (rank, alpha) in enumerate(MAIN_RANK_ALPHA_PAIRS):
                for j, lr in enumerate(MAIN_LEARNING_RATES):

                    cell = per_cell[
                        (per_cell["dataset"] == dataset)
                        & (per_cell["lora_target"] == target)
                        & (per_cell["rank"] == rank)
                        & (per_cell["alpha"] == alpha)
                        & (per_cell["learning_rate"] == lr)
                    ]

                    if cell.empty:
                        continue

                    n_collapsed = int(cell["n_collapsed"].iloc[0])
                    n = int(cell["n"].iloc[0])
                    col = d * group_stride + j

                    counts[i, col] = n_collapsed
                    n_seeds[i, col] = n

                    # Cells without collapse stay blank (NaN).
                    if n_collapsed > 0:
                        matrix[i, col] = n_collapsed / n

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        for i in range(n_rows):
            for col in range(n_cols):

                if n_seeds[i, col] == 0 or counts[i, col] == 0:
                    continue

                text_color = annot_color(matrix[i, col], cmap, norm)

                # Numerator only, to keep the narrow columns readable; a
                # border marks the n=10 cells instead.
                if n_seeds[i, col] != 3:
                    ax.add_patch(Rectangle(
                        (col - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor=text_color, linewidth=1.8,
                    ))

                ax.text(
                    col, i, str(counts[i, col]), ha="center", va="center",
                    color=text_color, fontsize=6,
                )

        ax.set_title(pretty(target), pad=20)

        if col_idx == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([rf"$r$={r:g}, $\alpha$={a:g}" for r, a in MAIN_RANK_ALPHA_PAIRS], fontsize=6)

        data_cols = [d * group_stride + j for d in range(n_datasets) for j in range(n_lr)]
        ax.set_xticks(data_cols)
        ax.set_xticklabels(
            [fmt_lr(lr) for _ in range(n_datasets) for lr in MAIN_LEARNING_RATES],
            fontsize=5, rotation=90,
        )
        ax.set_xlim(-0.5, n_cols - 0.5)

        for d, dataset in enumerate(MAIN_DATASETS):
            group_center = d * group_stride + (n_lr - 1) / 2
            ax.annotate(
                MAIN_DATASET_LABELS[dataset], xy=(group_center, -0.5),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5,
                annotation_clip=False,
            )

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.5, label="0/3"),
        Patch(facecolor=cmap(norm(1 / 3)), label="1/3"),
        Patch(facecolor=cmap(norm(2 / 3)), label="2/3"),
        Patch(facecolor=cmap(norm(1.0)), label="3/3"),
        Patch(facecolor="0.85", edgecolor="black", linewidth=1.8, label="risky-seed rerun, n=10"),
    ]
    # Larger font and handles than the defaults, since this legend is
    # the key to the whole figure.
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=5,
        frameon=False, title="Seeds collapsed (of 3, unless bordered)",
        fontsize=8.5, title_fontsize=9.5, handlelength=2.4, handleheight=1.6,
    )

    fig.supxlabel(pretty("learning_rate"))

    return fig, "instability_grid"


def fig_6_1a(df):
    """Plot peak and typical performance per strategy as dumbbells.

    One panel per dataset (2x2): a circle for the mean over the
    strategy's core grid and a star for the best configuration's seed
    mean with a +-1 SD bar. Uses CORE_BLOCKS only, so all strategies are
    compared on the same design; FFT has 3 configurations, each LoRA
    target 30.

    Shows that no strategy wins at peak on every dataset, and that qv
    stands out by its mean, not its peak. The zero-shot mean is shown as
    corner text so the y-axis can zoom in on the strategies.
    """
    working = _core_strategy_data(df)
    plot_strategies = [s for s in STRATEGY_ORDER if s != "zeroshot"]

    # 2x2 instead of 1x4, for wider panels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.92),
        sharey=False,
    )

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        sub = working[working["dataset"] == dataset]

        zeroshot_mean = sub.loc[sub["strategy"] == "zeroshot", "metric"].mean()

        # Lowest plotted point per edge strategy, used below to place
        # the zero-shot annotation under the edge with more room.
        edge_low = {}

        for x_idx, strategy in enumerate(plot_strategies):

            strat_sub = sub[sub["strategy"] == strategy]

            if strat_sub.empty:
                continue

            mean_val = strat_sub["metric"].mean()
            color = STRATEGY_COLORS[strategy]

            config_cols = ["learning_rate"] if strategy == "fft" else ["rank", "alpha", "learning_rate"]
            config_stats = strat_sub.groupby(config_cols)["metric"].agg(["mean", "std"])
            best_config = config_stats.loc[config_stats["mean"].idxmax()]
            best_mean, best_sd = best_config["mean"], best_config["std"]

            if x_idx in (0, len(plot_strategies) - 1):
                edge_low[x_idx] = min(mean_val, best_mean - best_sd)

            ax.plot([x_idx, x_idx], [mean_val, best_mean], color=color, linewidth=1.8, zorder=1)
            ax.errorbar(
                [x_idx], [best_mean], yerr=[[best_sd]], color=color,
                capsize=2, linewidth=0.8, zorder=2,
            )
            ax.scatter(
                [x_idx], [mean_val], marker="o", color=color, s=55, zorder=2, linewidths=0,
            )
            ax.scatter(
                [x_idx], [best_mean], marker="*", color=color, s=110, zorder=3,
                edgecolors="black", linewidths=0.5,
            )

        # Annotate under the edge strategy with the higher low point.
        left_low = edge_low.get(0, float("inf"))
        right_low = edge_low.get(len(plot_strategies) - 1, float("inf"))
        corner_x, ha = (0.03, "left") if left_low >= right_low else (0.97, "right")
        ax.text(
            corner_x, 0.03, f"Zero-shot mean: {zeroshot_mean:.3f}",
            ha=ha, va="bottom", fontsize=5.5, transform=ax.transAxes,
        )

        ax.set_xticks(range(len(plot_strategies)))
        ax.set_xticklabels([pretty(s) for s in plot_strategies], rotation=45, ha="right")
        ax.set_xlim(-0.5, len(plot_strategies) - 0.5)
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(METRIC_NAMES[dataset])

    legend_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="", markersize=6, label="Grid mean"),
        Line2D([0], [0], marker="*", color="black", linestyle="", markersize=9, label="Best config ($\\pm$ seed SD)"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=2,
        frameon=False,
    )

    return fig, "strategy_peak_vs_mean"


def _axis_break_bounds(values, min_gap=0.15, pad_frac=0.15):
    """Return y-bounds for a broken axis, or None if not needed.

    Finds the largest gap in the sorted values. If it is at least
    `min_gap` wide, returns (bottom_top, top_bottom), padded inward by
    `pad_frac` of the gap; otherwise the axis stays unbroken.
    """
    values = sorted(values)

    if len(values) < 2:
        return None

    gap_size, lo, hi = max((values[i + 1] - values[i], values[i], values[i + 1]) for i in range(len(values) - 1))

    if gap_size < min_gap:
        return None

    pad = gap_size * pad_frac

    return (lo + pad, hi - pad)


def _broken_panel(fig, spec, break_bounds, sharex=None, sharey_top=None, sharey_bottom=None, height_ratios=(1, 1)):
    """Create one panel in `spec`, broken if `break_bounds` is given.

    Returns (ax_top, ax_bottom); ax_bottom is None for an unbroken
    panel. `height_ratios` sets the sizes of the two segments.
    """
    if break_bounds is None:
        ax = fig.add_subplot(spec, sharex=sharex, sharey=sharey_top)
        return ax, None

    inner = spec.subgridspec(2, 1, height_ratios=height_ratios, hspace=0.08)
    ax_top = fig.add_subplot(inner[0], sharex=sharex, sharey=sharey_top)
    ax_bottom = fig.add_subplot(inner[1], sharex=ax_top, sharey=sharey_bottom)

    bottom_top, top_bottom = break_bounds
    ax_top.set_ylim(top_bottom, None)
    ax_bottom.set_ylim(None, bottom_top)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    d = 0.4
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
        color="black", mec="black", mew=1, clip_on=False,
    )
    ax_top.plot([0], [0], transform=ax_top.transAxes, **break_kwargs)
    ax_bottom.plot([0], [1], transform=ax_bottom.transAxes, **break_kwargs)

    return ax_top, ax_bottom


def fig_6_1e(df):
    """Plot the distribution of configuration means per strategy.

    One panel per dataset, one strip per strategy with its median and
    IQR. Points are per-configuration seed means (30 per LoRA target,
    3 for FFT, marked "n=3" with a hatched band). Uses CORE_BLOCKS only,
    so unequal seed counts cannot distort the distributions.

    The low tails are the point, so the y-axis is not clipped. Where the
    values form two separate clusters (_axis_break_bounds), the panel is
    broken into two axes.
    """
    working = _core_strategy_data(df)
    plot_strategies = ["fft", "qv", "attention", "full"]

    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85))
    gs = fig.add_gridspec(2, len(MAIN_DATASETS), height_ratios=[1, 1], hspace=0.08)

    for col_idx, dataset in enumerate(MAIN_DATASETS):

        sub = working[working["dataset"] == dataset]

        per_strategy_means = {}
        all_vals = []

        for strategy in plot_strategies:

            strat_sub = sub[sub["strategy"] == strategy]
            config_cols = ["learning_rate"] if strategy == "fft" else ["rank", "alpha", "learning_rate"]
            means = strat_sub.groupby(config_cols)["metric"].mean().values

            per_strategy_means[strategy] = means
            all_vals.extend(means.tolist())

        zeroshot_mean = sub.loc[sub["strategy"] == "zeroshot", "metric"].mean()
        break_bounds = _axis_break_bounds(all_vals)

        if break_bounds is not None:
            ax_top = fig.add_subplot(gs[0, col_idx])
            ax_bottom = fig.add_subplot(gs[1, col_idx], sharex=ax_top)
            panel_axes = [ax_top, ax_bottom]
        else:
            ax_single = fig.add_subplot(gs[:, col_idx])
            panel_axes = [ax_single]

        for ax in panel_axes:

            for x_idx, strategy in enumerate(plot_strategies):

                means = per_strategy_means[strategy]
                color = STRATEGY_COLORS[strategy]

                if strategy == "fft":
                    ax.axvspan(
                        x_idx - 0.4, x_idx + 0.4, facecolor="0.5", alpha=0.12,
                        hatch="//", edgecolor="0.5", linewidth=0, zorder=0,
                    )

                jitter = rng.uniform(-0.18, 0.18, size=len(means))
                ax.scatter(
                    x_idx + jitter, means, color=color, s=14, alpha=0.75,
                    linewidths=0, zorder=2,
                )

                q1, median, q3 = np.percentile(means, [25, 50, 75])
                ax.plot([x_idx - 0.28, x_idx + 0.28], [median, median], color="black", linewidth=1.6, zorder=3)
                ax.plot([x_idx, x_idx], [q1, q3], color="black", linewidth=0.8, alpha=0.6, zorder=1)

            ax.axhline(
                zeroshot_mean, color=STRATEGY_COLORS["zeroshot"],
                linestyle="--", linewidth=1.2, alpha=0.7, zorder=0,
            )

            ax.set_xlim(-0.5, len(plot_strategies) - 0.5)

        if break_bounds is not None:

            bottom_top, top_bottom = break_bounds
            ax_top.set_ylim(top_bottom, None)
            ax_bottom.set_ylim(None, bottom_top)

            ax_top.spines["bottom"].set_visible(False)
            ax_bottom.spines["top"].set_visible(False)
            ax_top.tick_params(labelbottom=False, bottom=False)

            # Standard broken-axis diagonal marks, drawn in each axes'
            # own fraction coordinates so they stay put under resizing.
            d = 0.4
            break_kwargs = dict(
                marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
                color="black", mec="black", mew=1, clip_on=False,
            )
            ax_top.plot([0], [0], transform=ax_top.transAxes, **break_kwargs)
            ax_bottom.plot([0], [1], transform=ax_bottom.transAxes, **break_kwargs)

            ax_top.set_title(MAIN_DATASET_LABELS[dataset])
            top_ax_for_labels, bottom_ax_for_labels = ax_top, ax_bottom

        else:

            ax_single.set_title(MAIN_DATASET_LABELS[dataset])
            top_ax_for_labels = bottom_ax_for_labels = ax_single

        bottom_ax_for_labels.set_xticks(range(len(plot_strategies)))
        bottom_ax_for_labels.set_xticklabels([pretty(s) for s in plot_strategies], rotation=45, ha="right")

        if bottom_ax_for_labels is not top_ax_for_labels:
            top_ax_for_labels.tick_params(labelbottom=False)

        # Metric name on the top (or only) segment.
        top_ax_for_labels.set_ylabel(METRIC_NAMES[dataset])

        # "n=3" on every segment, since FFT's points can fall into both.
        for ax in panel_axes:
            ax.text(
                0, 0.04, "n=3", ha="center", va="bottom", fontsize=5.5,
                transform=ax.get_xaxis_transform(),
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="0.3", linestyle="", markersize=5, label="Per-config mean"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Median"),
        Line2D([0], [0], color="black", linewidth=0.8, alpha=0.6, label="IQR"),
        Line2D([0], [0], color=STRATEGY_COLORS["zeroshot"], linestyle="--", linewidth=1.2, label="Zero-shot mean"),
        Patch(facecolor="0.5", alpha=0.12, hatch="//", edgecolor="0.5", label=f"FFT (n=3, vs. n={MAIN_N_CONFIGS})"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=3,
        frameon=False,
    )

    return fig, "strategy_score_distribution"


def fig_6_1b(df):
    """Plot FFT seed stability as a strip plot per learning rate.

    Uses fft_wd0.1 plus its extra seeds (n=10 per cell on every
    dataset); collapsed runs (is_collapsed()) are drawn as a black X.
    n is annotated once per panel, since all groups share it. MNLI gets
    an inset zoomed to its healthy range.
    """
    stability = _fft_stability_data(df)

    eligible = pd.concat([stability, df[df["block"] == "zeroshot"]])
    collapsed = is_collapsed(eligible)
    stability = stability.assign(_collapsed=collapsed.loc[stability.index])
    zeroshot_means = _zeroshot_means(df)

    # 2x2 instead of 1x4, for wider panels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.84),
        sharey=False,
    )

    rng = np.random.default_rng(0)

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        sub = stability[stability["dataset"] == dataset]

        # Main panel only; in the inset the zero-shot line would undo
        # the zoom.
        _draw_zeroshot_line(ax, zeroshot_means[dataset])

        if dataset == "mnli":
            inset = ax.inset_axes([0.42, 0.34, 0.53, 0.4])
            inset.set_xticks(range(len(FFT_LEARNING_RATES)))
            inset.set_xticklabels([])
            inset.tick_params(labelsize=6, pad=1.5)
            inset.set_xlim(-0.5, len(FFT_LEARNING_RATES) - 0.5)
        else:
            inset = None

        group_ns = []

        for x_idx, lr in enumerate(FFT_LEARNING_RATES):

            lr_sub = sub[sub["learning_rate"] == lr]
            n = len(lr_sub)
            group_ns.append(n)

            jitter = pd.Series(rng.uniform(-0.15, 0.15, size=n), index=lr_sub.index)

            healthy = lr_sub[~lr_sub["_collapsed"]]
            collapsed_pts = lr_sub[lr_sub["_collapsed"]]

            ax.scatter(
                x_idx + jitter.loc[healthy.index], healthy["metric"],
                color=STRATEGY_COLORS["fft"], s=16, alpha=0.7, linewidths=0,
            )
            ax.scatter(
                x_idx + jitter.loc[collapsed_pts.index], collapsed_pts["metric"],
                color="black", marker="x", s=24, linewidths=1.2,
            )

            if inset is not None:
                inset.scatter(
                    x_idx + jitter.loc[healthy.index], healthy["metric"],
                    color=STRATEGY_COLORS["fft"], s=8, alpha=0.7, linewidths=0,
                )

        # One n annotation per panel, since all groups have the same n.
        assert len(set(group_ns)) == 1, (
            f"fig_6_1b assumes a uniform n per learning-rate group, got {group_ns}"
        )
        ax.text(
            0.03, 0.97, f"n={group_ns[0]} per point", ha="left", va="top",
            fontsize=6, transform=ax.transAxes,
        )

        ax.set_xticks(range(len(FFT_LEARNING_RATES)))
        ax.set_xticklabels([fmt_lr(lr) for lr in FFT_LEARNING_RATES], rotation=30, ha="right")
        ax.set_xlim(-0.5, len(FFT_LEARNING_RATES) - 0.5)
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(METRIC_NAMES[dataset])

        if inset is not None:
            healthy_all = sub[~sub["_collapsed"]]["metric"]
            lo, hi = healthy_all.min(), healthy_all.max()
            pad = (hi - lo) * 0.15 or 0.005
            inset.set_ylim(lo - pad, hi + pad)

    legend_handles = [
        Line2D([0], [0], marker="o", color=STRATEGY_COLORS["fft"], linestyle="", markersize=5, label="Healthy seed"),
        Line2D([0], [0], marker="x", color="black", linestyle="", markersize=5, label="Collapsed seed"),
        Line2D([0], [0], color=STRATEGY_COLORS["zeroshot"], linestyle="--", linewidth=1.2, label="Zero-shot mean"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=3,
        frameon=False,
    )

    # Every panel's x-axis is the learning rate.
    fig.supxlabel(pretty("learning_rate"))

    return fig, "fft_seed_stability"


def fig_6_1c(df):
    """Plot collapse rates per strategy and dataset with Wilson CIs.

    The CI width shows how much n=3 and n=10 measurements differ: 0/3
    and 0/10 have the same rate but very different certainty. FFT has
    an n=3 point (core block) and an n=10 point (wd 0.1 diagnostic).
    Attention and full are split into their n=3 cells and the n=10
    risky-seed cells, detected from the seed count per cell. Missing
    risky cells (e.g. GSM8K) are left as gaps.

    The CI treats all cells of a group as exchangeable trials, which is
    an approximation, since the risk concentrates at high rank and LR.
    Tick labels state the seeds per cell; the CI uses the pooled count.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    stability = _fft_stability_data(df)
    risky = df[df["block"].isin(RISKY_SEED_BLOCKS)]

    eligible = attach_strategy(pd.concat([core, stability, risky]))
    collapsed = is_collapsed(eligible)
    eligible = eligible.assign(_collapsed=collapsed)

    main_cell_cols = ["dataset", "rank", "alpha", "learning_rate"]

    def _split_by_cell_depth(sub):
        """Split `sub` into cells with <= 3 seeds and cells with more.

        Based on each cell's own row count, not a fixed cell list.
        """
        cell_size = sub.groupby(main_cell_cols)["seed"].transform("count")

        return sub[cell_size <= 3], sub[cell_size > 3]

    attn_rest, attn_risky = _split_by_cell_depth(eligible[eligible["strategy"] == "attention"])
    full_rest, full_risky = _split_by_cell_depth(eligible[eligible["strategy"] == "full"])

    # cell_cols define one cell, for the seeds-per-cell tick labels.
    # Markers: circle for n=3, diamond for n=10.
    lr_cols = ["learning_rate"]
    # (strategy, suffix or None, data, color, marker, cell_cols)
    bar_specs = [
        ("fft", None, eligible[eligible["block"] == "fft"], STRATEGY_COLORS["fft"], "o", lr_cols),
        ("fft", None, eligible[eligible["block"] == "fft_stability"], STRATEGY_COLORS["fft"], "D", lr_cols),
        ("qv", None, eligible[eligible["strategy"] == "qv"], STRATEGY_COLORS["qv"], "o", main_cell_cols),
        ("attention", "rest", attn_rest, STRATEGY_COLORS["attention"], "o", main_cell_cols),
        ("attention", "risky", attn_risky, STRATEGY_COLORS["attention"], "D", main_cell_cols),
        ("full", "rest", full_rest, STRATEGY_COLORS["full"], "o", main_cell_cols),
        ("full", "risky", full_risky, STRATEGY_COLORS["full"], "D", main_cell_cols),
    ]

    fig, axes = plt.subplots(
        1, len(MAIN_DATASETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.6),
        sharey=True,
    )

    max_upper = 0.0

    for ax, dataset in zip(axes, MAIN_DATASETS):

        tick_labels = []

        for x_idx, (base, suffix, group, color, marker, cell_cols) in enumerate(bar_specs):

            label = pretty(base) if suffix is None else f"{pretty(base)} ({suffix})"

            cell = group[group["dataset"] == dataset]
            n = len(cell)

            if n == 0:
                # No risky cells here (e.g. GSM8K): absent, not zero.
                tick_labels.append(f"{label} (n/a)")
                continue

            k = int(cell["_collapsed"].sum())
            rate = k / n
            lower, upper = _wilson_ci(k, n)
            max_upper = max(max_upper, upper)

            n_cells = cell.groupby(cell_cols).ngroups
            assert n % n_cells == 0, (
                f"fig_6_1c: n={n} is not a whole multiple of n_cells={n_cells} for {label}/{dataset}"
            )
            seeds_per_cell = n // n_cells

            ax.scatter(
                x_idx, rate, color=color, marker=marker, s=32, zorder=3,
                edgecolors="black", linewidths=0.4,
            )

            # max(0, ...) guards against float error in _wilson_ci at 0
            # or 1, which errorbar would reject as a negative yerr.
            ax.errorbar(
                x_idx, rate, yerr=[[max(0.0, rate - lower)], [max(0.0, upper - rate)]],
                color="black", capsize=2, linewidth=0.8, zorder=2,
            )

            tick_labels.append(f"{label} (n={seeds_per_cell}/cell)")

        ax.set_xticks(range(len(bar_specs)))
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=90, ha="center")
        ax.set_xlim(-0.5, len(bar_specs) - 0.5)
        ax.set_title(MAIN_DATASET_LABELS[dataset])

    axes[0].set_ylim(0, max_upper * 1.15)
    axes[0].set_ylabel("Share of seeds collapsed")

    return fig, "collapse_rate_by_strategy"


# fig_6_1d is a LaTeX table, see src/analysis/tables.py.


# Chapter 6.2 efficiency figures. Cost columns use the categorical
# palette, never the strategy colors.

COST_COLUMNS = ["runtime_s", "peak_mem_gb", "trainable_params"]
COST_COLORS = categorical_color(COST_COLUMNS)


def fig_6_2a(df):
    """Plot trainable parameters vs. metric per configuration.

    One panel per dataset; x is the log parameter count, y the seed mean
    per configuration, color the target, FFT a star. The stepped line is
    the Pareto frontier over all strategies. Across about four orders of
    magnitude in parameters the metric barely moves; above qv mostly the
    spread grows. Markers have a thin edge so overlapping points stay
    distinguishable.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    lora = core[core["adaptation"] == "lora"]
    fft = core[core["adaptation"] == "fft"]
    zeroshot_means = _zeroshot_means(df)

    lora_means = (
        lora.groupby(["dataset", "lora_target", "rank", "alpha", "learning_rate"], as_index=False)
        .agg(metric=("metric", "mean"), trainable_params=("trainable_params", "first"))
    )
    fft_means = (
        fft.groupby(["dataset", "learning_rate"], as_index=False)
        .agg(metric=("metric", "mean"), trainable_params=("trainable_params", "first"))
    )

    # 2x2 instead of 1x4, for wider panels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.78),
        sharey=False,
    )

    # A marker shape per target in addition to color, so the targets
    # stay distinguishable in grayscale.
    TARGET_MARKERS = {"qv": "o", "attention": "s", "full": "^"}

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        lora_sub = lora_means[lora_means["dataset"] == dataset]
        fft_sub = fft_means[fft_means["dataset"] == dataset]

        _draw_zeroshot_line(ax, zeroshot_means[dataset])

        for target in MAIN_TARGETS:

            target_sub = lora_sub[lora_sub["lora_target"] == target]

            ax.scatter(
                target_sub["trainable_params"], target_sub["metric"],
                marker=TARGET_MARKERS[target], color=STRATEGY_COLORS[target],
                s=14, alpha=0.7, edgecolors="black", linewidths=0.3,
            )

        ax.scatter(
            fft_sub["trainable_params"], fft_sub["metric"],
            color=STRATEGY_COLORS["fft"], marker="*", s=70, linewidths=0,
            zorder=5,
        )

        # Pareto frontier over all strategies: the best metric at or
        # under each parameter budget.
        combined = pd.concat([
            lora_sub[["trainable_params", "metric"]],
            fft_sub[["trainable_params", "metric"]],
        ]).sort_values("trainable_params").reset_index(drop=True)

        running_max = combined["metric"].cummax()
        is_record = running_max != running_max.shift(1)
        frontier = combined[is_record]

        ax.step(
            frontier["trainable_params"], frontier["metric"],
            where="post", color="black", linewidth=2.2, zorder=6,
        )

        ax.set_xscale("log")
        # Explicit decade ticks; the default locator shows too few here.
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(METRIC_NAMES[dataset])

        # Add a top margin so the frontier's maximum does not touch the
        # upper spine.
        y_bottom, y_top = ax.get_ylim()
        ax.set_ylim(y_bottom, y_top + (y_top - y_bottom) * 0.08)

    # One shared x-label; four per-panel labels would overlap.
    fig.supxlabel("Trainable parameters")

    legend_handles = [
        Line2D([0], [0], marker=TARGET_MARKERS[t], color=STRATEGY_COLORS[t], linestyle="", markersize=5, label=pretty(t))
        for t in MAIN_TARGETS
    ] + [
        Line2D([0], [0], marker="*", color=STRATEGY_COLORS["fft"], linestyle="", markersize=8, label=pretty("fft")),
        Line2D([0], [0], color="black", linewidth=1, label="Pareto frontier"),
        Line2D([0], [0], color=STRATEGY_COLORS["zeroshot"], linestyle="--", linewidth=1.2, label="Zero-shot mean"),
    ]
    # Tighter spacing and a smaller font keep the legend within the text
    # width; six columns fit all entries in one row.
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=6,
        frameon=False, fontsize=6.5, handletextpad=0.4,
        columnspacing=1.0, handlelength=1.5,
    )

    return fig, "params_vs_performance"


def fig_6_2b(df):
    """Plot runtime (top row) and peak memory (bottom row) per strategy.

    Bars with +-1 SD over each strategy's core grid, one column per
    dataset, y-axes shared per row. Zero-shot has no runtime (it never
    trains), so its runtime slot stays empty, but it does have a memory
    bar. Runtime depends on the hardware and is only comparable within
    this study.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    core = core[core["adaptation"] != "none"]
    core = attach_strategy(core)

    zeroshot = df[df["block"] == "zeroshot"]

    strategies = ["zeroshot", "fft", "qv", "attention", "full"]
    panels = [("runtime_s", "Runtime (s)"), ("peak_mem_gb", "Peak memory (GB)")]

    fig, axes = plt.subplots(
        2, len(MAIN_DATASETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.68),
        sharex=True, sharey="row",
    )

    for row_idx, (column, ylabel) in enumerate(panels):
        for col_idx, dataset in enumerate(MAIN_DATASETS):

            ax = axes[row_idx, col_idx]
            sub = core[core["dataset"] == dataset]
            zeroshot_sub = zeroshot[zeroshot["dataset"] == dataset]

            means = []
            stds = []
            colors = []

            for s in strategies:
                # Zero-shot runtime is all NaN, so no bar is drawn.
                values = zeroshot_sub[column] if s == "zeroshot" else sub.loc[sub["strategy"] == s, column]
                means.append(values.mean())
                stds.append(values.std())
                colors.append(STRATEGY_COLORS[s])

            ax.bar(
                range(len(strategies)), means, yerr=stds, color=colors,
                capsize=2, error_kw={"linewidth": 0.8},
            )

            if row_idx == 0:
                ax.set_title(MAIN_DATASET_LABELS[dataset])

            # A y-label on every panel, so each can be read on its own.
            ax.set_ylabel(ylabel)

            ax.set_xticks(range(len(strategies)))

            if row_idx == 1:
                ax.set_xticklabels([pretty(s) for s in strategies], rotation=45, ha="right")
            else:
                ax.tick_params(labelbottom=False)

    return fig, "runtime_memory"


def fig_6_2c(df):
    """Plot runtime, memory and parameters relative to FFT (= 1.0).

    Ratios are computed per dataset and then averaged (mean +- SD over
    datasets); raw values are never averaged across datasets. Trainable
    parameters get their own lower panel, as their ratios are about
    100x smaller.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    core = core[core["adaptation"] != "none"]
    core = attach_strategy(core)

    rows = []

    for dataset in MAIN_DATASETS:

        sub = core[core["dataset"] == dataset]
        fft_means = sub.loc[sub["strategy"] == "fft", COST_COLUMNS].mean()

        for target in MAIN_TARGETS:

            target_means = sub.loc[sub["strategy"] == target, COST_COLUMNS].mean()

            for cost in COST_COLUMNS:
                rows.append({
                    "dataset": dataset, "target": target, "cost": cost,
                    "ratio": target_means[cost] / fft_means[cost],
                })

    ratios = pd.DataFrame(rows)
    summary = ratios.groupby(["target", "cost"])["ratio"].agg(["mean", "std"]).reset_index()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
        height_ratios=[2.2, 1],
    )

    width = 0.35
    x_positions = list(range(len(MAIN_TARGETS)))
    top_costs = ["runtime_s", "peak_mem_gb"]

    for i, cost in enumerate(top_costs):

        offsets = [x + (i - 0.5) * width for x in x_positions]

        means = [
            summary.loc[(summary["target"] == t) & (summary["cost"] == cost), "mean"].iloc[0]
            for t in MAIN_TARGETS
        ]
        stds = [
            summary.loc[(summary["target"] == t) & (summary["cost"] == cost), "std"].iloc[0]
            for t in MAIN_TARGETS
        ]

        ax_top.bar(
            offsets, means, width=width, yerr=stds, color=COST_COLORS[cost],
            capsize=2, label=pretty(cost), edgecolor="black", linewidth=0.3,
        )

        for x, mean, std in zip(offsets, means, stds):
            ax_top.text(
                x, mean + std + 0.03, f"{mean * 100:.1f}%",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

    # Neutral grey: the line marks FFT's level (1.0), and the FFT color
    # would look like another bar series.
    ax_top.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax_top.set_ylim(0, 1.35)

    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels([pretty(t) for t in MAIN_TARGETS])
    ax_top.set_ylabel("Ratio to FFT")
    ax_top.legend(frameon=False, fontsize=6)

    # Bottom panel: trainable parameters alone, on their own axis.
    means = [
        summary.loc[(summary["target"] == t) & (summary["cost"] == "trainable_params"), "mean"].iloc[0]
        for t in MAIN_TARGETS
    ]
    stds = [
        summary.loc[(summary["target"] == t) & (summary["cost"] == "trainable_params"), "std"].iloc[0]
        for t in MAIN_TARGETS
    ]

    ax_bottom.bar(
        x_positions, means, width=0.5, yerr=stds, color=COST_COLORS["trainable_params"],
        capsize=2, label=pretty("trainable_params"), edgecolor="black", linewidth=0.3,
    )

    top_of_range = max(m + s for m, s in zip(means, stds))
    for x, mean, std in zip(x_positions, means, stds):
        ax_bottom.text(
            x, mean + std + top_of_range * 0.08, f"{mean * 100:.2f}%",
            ha="center", va="bottom", fontsize=6.5,
        )

    ax_bottom.set_ylim(0, top_of_range * 1.5)
    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([pretty(t) for t in MAIN_TARGETS])
    ax_bottom.set_ylabel("Ratio to FFT")
    # Two decimals, so ticks such as 0.25% are not rounded unevenly.
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax_bottom.legend(frameon=False, fontsize=6)

    return fig, "relative_cost_vs_fft"



def fig_6_4(df):
    """Plot transfer regret between datasets as three 4x4 heatmaps.

    Rows are source datasets, columns targets; the diagonal is zero.
    (a) transfers the full best configuration, (b) the same with the
    learning rate re-tuned on the target, and (c) = a - b, the regret
    caused by the learning rate alone (never negative, since the
    transferred rate is one of the candidates in b).

    One shared color scale and colorbar, so color is proportional to
    regret everywhere; the exact values are annotated. Panels a and b
    sit side by side with c centered below. They are placed manually in
    inches with the layout engine disabled, which gives larger panels
    than constrained_layout.
    """
    cmap = plt.get_cmap("OrRd")

    matrices = {}
    for retune_lr, key in [(False, "A"), (True, "B")]:
        m = np.zeros((len(MAIN_DATASETS), len(MAIN_DATASETS)))
        for i, source in enumerate(MAIN_DATASETS):
            for j, target in enumerate(MAIN_DATASETS):
                m[i, j] = regret(df, source, target, retune_lr=retune_lr)
        matrices[key] = m

    matrices["C"] = matrices["A"] - matrices["B"]

    # One shared norm, so color is proportional to regret in all panels.
    global_max = max(m.max() for m in matrices.values())
    norm = plt.Normalize(vmin=0, vmax=global_max)

    # Bold lowercase panel letters, as referenced in the caption.
    panel_titles = {
        "A": r"$\mathbf{a}$ Full config transferred",
        "B": r"$\mathbf{b}$ LR retuned on target",
        "C": r"$\mathbf{c}$ a $-$ b (LR-attributable)",
    }

    # Margins in inches, found by rendering; add_axes() reserves no room
    # for labels, so smaller values clip text.
    left_label_margin_in = 0.55
    inter_col_gap_in = 0.16
    cbar_gap_in = 0.09
    cbar_width_in = 0.13
    label_margin_in = 0.55

    title_h_in = 0.22
    row_gap_in = 0.16
    bottom_margin_in = 0.58
    top_margin_in = 0.08

    # Fixed total height, well below MAX_FIG_HEIGHT_IN, which sets the
    # panel size.
    fig_height_in = 4.3
    panel_size_in = (
        fig_height_in - top_margin_in - 2 * title_h_in - row_gap_in - bottom_margin_in
    ) / 2

    row_content_width_in = (
        left_label_margin_in + panel_size_in + inter_col_gap_in + panel_size_in
        + cbar_gap_in + cbar_width_in + label_margin_in
    )
    left_start_in = left_label_margin_in + (TEXTWIDTH_IN - row_content_width_in) / 2

    fig = plt.figure(figsize=(TEXTWIDTH_IN, fig_height_in))
    fig.set_layout_engine(None)

    labels = [MAIN_DATASET_LABELS[d] for d in MAIN_DATASETS]

    row1_y_in = fig_height_in - top_margin_in - title_h_in - panel_size_in
    row2_y_in = row1_y_in - row_gap_in - title_h_in - panel_size_in

    # (x0_in, y0_in) per panel: a and b on top, c centered below.
    panel_x0_in = {
        "A": left_start_in,
        "B": left_start_in + panel_size_in + inter_col_gap_in,
        "C": left_start_in + (panel_size_in + inter_col_gap_in) / 2,
    }
    panel_y0_in = {"A": row1_y_in, "B": row1_y_in, "C": row2_y_in}

    for key in ["A", "B", "C"]:

        ax = fig.add_axes([
            panel_x0_in[key] / TEXTWIDTH_IN, panel_y0_in[key] / fig_height_in,
            panel_size_in / TEXTWIDTH_IN, panel_size_in / fig_height_in,
        ])

        matrix = matrices[key]

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")

        for i in range(len(MAIN_DATASETS)):
            for j in range(len(MAIN_DATASETS)):

                text_color = annot_color(matrix[i, j], cmap, norm)

                ax.text(
                    j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color=text_color, fontsize=7.5,
                )

        ax.set_xticks(range(len(MAIN_DATASETS)))
        ax.set_yticks(range(len(MAIN_DATASETS)))
        ax.set_title(panel_titles[key], fontsize=9)

        # Row labels on panels a and c (b sits next to a); column labels
        # only on c at the bottom.
        if key in ("A", "C"):
            ax.set_yticklabels(labels, fontsize=7.5)
            ax.set_ylabel("Source")
        else:
            ax.set_yticklabels([])

        if key == "C":
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
            ax.set_xlabel("Target")
        else:
            ax.set_xticklabels([])
            ax.tick_params(labelbottom=False)

    # One tall colorbar for all three panels.
    cax = fig.add_axes([
        (panel_x0_in["B"] + panel_size_in + cbar_gap_in) / TEXTWIDTH_IN,
        row2_y_in / fig_height_in,
        cbar_width_in / TEXTWIDTH_IN,
        (row1_y_in + panel_size_in - row2_y_in) / fig_height_in,
    ])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Regret", fontsize=8)

    return fig, "transfer_regret"


def fig_6_4d(df):
    """Plot the percentile rank of each transferred configuration.

    Same 4x4 layout as fig_6_4; the cell value is the percentile rank
    (0-100) of the source's best configuration among the target's 90
    configurations. Percentiles are scale-free, so one color scale is
    shared. Cells are square; annotations get an outline for contrast.
    """
    matrix = np.zeros((len(MAIN_DATASETS), len(MAIN_DATASETS)))

    for i, source in enumerate(MAIN_DATASETS):
        for j, target in enumerate(MAIN_DATASETS):
            matrix[i, j] = percentile_rank(df, source, target)

    # A shorter figure makes the square grid smaller (the width is fixed
    # by the size contract).
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.62))

    # A square axes box via set_box_aspect(), which works with
    # constrained_layout (unlike imshow's aspect="equal").
    ax.set_box_aspect(1)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=100)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label="Percentile rank on target", fraction=0.046, pad=0.04)

    for i in range(len(MAIN_DATASETS)):
        for j in range(len(MAIN_DATASETS)):

            text_color = annot_color(matrix[i, j], cmap, norm)
            stroke_color = "black" if text_color == "white" else "white"

            ax.text(
                j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                color=text_color, fontsize=6.5,
                path_effects=[patheffects.withStroke(linewidth=1.3, foreground=stroke_color)],
            )

    labels = [MAIN_DATASET_LABELS[d] for d in MAIN_DATASETS]

    ax.set_xticks(range(len(MAIN_DATASETS)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(MAIN_DATASETS)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")

    # constrained_layout centers the square axes, leaving empty space on
    # the left. After a first draw, shift the axes and colorbar left by
    # the same amount and freeze the layout so savefig() keeps it.
    fig.canvas.draw()

    main_pos = ax.get_position()
    cbar_pos = cbar.ax.get_position()

    target_left_margin = 0.22
    delta = target_left_margin - main_pos.x0

    ax.set_position([main_pos.x0 + delta, main_pos.y0, main_pos.width, main_pos.height])
    cbar.ax.set_position([cbar_pos.x0 + delta, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

    fig.set_layout_engine(None)

    return fig, "transfer_percentile_rank"


def fig_6_4e(df):
    """Plot the best configuration of each dataset, one row each.

    A dot colored by target at the best configuration's mean metric,
    annotated with the configuration. Every row spans [0, 1], a common
    scale for the bounded metrics (not a common meaning). No two
    datasets share the full winning configuration, although all four
    use lr 1e-4.
    """
    zeroshot_means = _zeroshot_means(df)

    fig, axes = plt.subplots(
        len(MAIN_DATASETS), 1,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.55),
    )

    for ax, dataset in zip(axes, MAIN_DATASETS):

        bc = best_config(df, dataset)
        target = bc["lora_target"]
        metric = bc["mean_metric"]

        # The score axis is horizontal here, so the zero-shot line is an
        # axvline in the usual style.
        ax.axvline(
            zeroshot_means[dataset], color=STRATEGY_COLORS["zeroshot"],
            linestyle="--", linewidth=1.2, alpha=0.7, zorder=0,
        )

        ax.scatter([metric], [0], color=STRATEGY_COLORS[target], s=60, zorder=3)

        label = (
            rf"{pretty(target)}, $r$={bc['rank']:.0f}, $\alpha$={bc['alpha']:.0f}, "
            f"lr={fmt_lr(bc['learning_rate'])}  (score={metric:.3f})"
        )
        ax.annotate(
            label, xy=(metric, 0), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=7,
        )

        ax.set_xlim(0, 1)
        ax.set_yticks([0])
        ax.set_yticklabels([MAIN_DATASET_LABELS[dataset]])
        ax.set_ylim(-1, 1)
        ax.tick_params(left=False)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Score (0-1; each row its own scale)")

    legend_handles = [
        Line2D([0], [0], color=STRATEGY_COLORS["zeroshot"], linestyle="--", linewidth=1.2, label="Zero-shot mean"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=1,
        frameon=False,
    )

    return fig, "best_config_per_dataset"


# The strategy colors reused for source datasets, in fig_6_4f only (to
# match fig_6_2a); not a general dataset color convention.
DATASET_TRANSFER_COLORS = dict(zip(
    MAIN_DATASETS,
    [STRATEGY_COLORS["qv"], STRATEGY_COLORS["attention"], STRATEGY_COLORS["full"], STRATEGY_COLORS["fft"]],
))


def fig_6_4f(df):
    """Plot transfer regret against a leave-one-out default per target.

    One panel per target dataset, each with its own y-axis. Circles are
    the regrets of the other datasets' best configurations (colored by
    source); the grey diamond is the leave-one-out default, chosen
    without the target's data. The shaded band marks the winner's seed
    SD (noise floor), the dashed line the grid median regret. Every
    point is annotated.

    The y-axis ends at 0.035 (state this in the caption); all plotted
    points lie below it, only the grid's tail is cut. Points are never
    pooled across datasets.
    """
    # 2x2 instead of 1x4, for wider panels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.94),
        sharex=False, sharey=False,
    )

    for ax, target in zip(axes.flat, MAIN_DATASETS):

        sources = [d for d in MAIN_DATASETS if d != target]
        transfer_regrets = [regret(df, source, target, retune_lr=False) for source in sources]

        loo_config, loo_regret = leave_one_out_default(df, target, MAIN_DATASETS)
        seed_sd = winner_seed_sd(df, target)
        grid_median = grid_median_regret(df, target)

        x_positions = list(range(len(sources) + 1))

        # Reference band/line drawn first (zorder 0-1), points on top.
        ax.axhspan(0, seed_sd, color="0.85", zorder=0)
        ax.axhline(grid_median, color="black", linestyle="--", linewidth=1, zorder=1)

        ax.scatter(
            x_positions[:-1], transfer_regrets, marker="o", s=32,
            color=[DATASET_TRANSFER_COLORS[s] for s in sources],
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        ax.scatter(
            [x_positions[-1]], [loo_regret], marker="D", s=32,
            color="0.5", edgecolors="black", linewidths=0.4, zorder=3,
        )

        # Horizontal labels with a 6pt offset, so they do not touch the
        # markers.
        for x, value in zip(x_positions, transfer_regrets + [loo_regret]):
            ax.annotate(
                f"{value:.3f}", xy=(x, value), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, zorder=4,
            )

        # No x-tick labels: color and legend identify the sources, the
        # diamond marks the default.
        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(sources) + 0.6)
        # Room above the highest point (about 0.030), so its annotation
        # does not touch the spine.
        ax.set_ylim(0, 0.035)
        ax.set_title(MAIN_DATASET_LABELS[target])

        # A y-label on every panel, since the 2x2 grid has no shared
        # y-axis; the scale caveat is in the caption.
        ax.set_ylabel("Regret")

    legend_handles = [
        Line2D([0], [0], marker="o", color=DATASET_TRANSFER_COLORS[d], linestyle="", markersize=5, label=MAIN_DATASET_LABELS[d])
        for d in MAIN_DATASETS
    ] + [
        Line2D([0], [0], marker="D", color="0.5", linestyle="", markersize=5, label="Leave-one-out default"),
        Patch(facecolor="0.85", edgecolor="none", label="Seed spread (0 to winner's SD)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Grid median regret"),
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=4,
        frameon=False, fontsize=6.5,
    )

    return fig, "transfer_vs_default"


# Chapter 6.6 Shapley figures. Contributors use one categorical color
# assignment in all Shapley figures, never the strategy colors.
CONTRIBUTOR_COLORS = categorical_color([
    "lora_target", "learning_rate", "rank_alpha",
    "rank", "scaling_ratio", "subset_size", "training_regime",
])


def _grouped_shapley_bars(shapley_df, contributors, dataset_labels, figsize, annotate_exact_zero=False):
    """Draw grouped Shapley bars (shared by fig_6_6a, 6_6c and 6_6d).

    With annotate_exact_zero, bars that are exactly 0.0 get a small "0"
    label, so a real zero is not mistaken for a missing value.
    """
    datasets = list(dataset_labels.keys())

    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / len(contributors)
    x_positions = list(range(len(datasets)))
    zero_positions = []

    for i, contributor in enumerate(contributors):

        offsets = [x + (i - (len(contributors) - 1) / 2) * width for x in x_positions]

        values = [
            shapley_df.loc[
                (shapley_df["dataset"] == d) & (shapley_df["contributor"] == contributor),
                "shapley_value",
            ].iloc[0]
            for d in datasets
        ]

        ax.bar(
            offsets, values, width=width, color=CONTRIBUTOR_COLORS[contributor],
            label=pretty(contributor), edgecolor="black", linewidth=0.3,
        )

        zero_positions.extend(x for x, value in zip(offsets, values) if value == 0.0)

    ax.axhline(0, color="black", linewidth=0.6)

    if annotate_exact_zero and zero_positions:
        # Offset relative to the y-range, since this helper serves
        # figures on different scales.
        _, y_top = ax.get_ylim()
        for x in zero_positions:
            ax.text(
                x, y_top * 0.02, "0 (exact)", ha="center", va="bottom",
                fontsize=5.5, rotation=90,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dataset_labels[d] for d in datasets])
    ax.set_ylabel("Shapley value")
    ax.legend(frameon=False, fontsize=6)

    return fig


def fig_6_6a(df):
    """Plot main-sweep Shapley values per dataset as grouped bars.

    Contributors: lora_target, learning_rate and rank_alpha (exact
    enumeration, see shapley.py). Values are comparable within one
    dataset, not in absolute size across datasets.
    """
    shapley_df, _ = all_main_sweep_attributions(df)

    fig = _grouped_shapley_bars(
        shapley_df, list(MAIN_CONTRIBUTORS.keys()), MAIN_DATASET_LABELS,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.45),
    )

    return fig, "shapley_main_sweep"


def fig_6_6b(df):
    """Plot main-sweep interaction heatmaps, one 3x3 panel per dataset.

    The diagonal holds the Shapley values, the off-diagonal cells the
    Grabisch-Roubens interactions. RdBu_r is symmetric around 0 and
    scaled per panel, since only within-panel comparisons are
    meaningful. Values are annotated x1000 (state this in the caption).
    """
    contributors = list(MAIN_CONTRIBUTORS.keys())
    n = len(contributors)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN),
    )

    cmap = plt.get_cmap("RdBu_r")

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        shapley, interactions = main_sweep_attribution(df, dataset)

        matrix = np.zeros((n, n))
        for i, ci in enumerate(contributors):
            matrix[i, i] = shapley[ci]
        for (ci, cj), value in interactions.items():
            i, j = contributors.index(ci), contributors.index(cj)
            matrix[i, j] = value
            matrix[j, i] = value

        vmax = np.abs(matrix).max()
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        for i in range(n):
            for j in range(n):

                text_color = annot_color(matrix[i, j], cmap, norm)

                ax.text(
                    j, i, f"{matrix[i, j] * 1000:.2f}", ha="center", va="center",
                    color=text_color, fontsize=6.5,
                )

        pretty_contributors = [pretty(c) for c in contributors]
        ax.set_xticks(range(n))
        ax.set_xticklabels(pretty_contributors, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(pretty_contributors, fontsize=6.5)
        ax.set_title(MAIN_DATASET_LABELS[dataset])

    # supxlabel reserves its own space; a fig.text outside the figure
    # would be clipped, since the canvas is never expanded.
    fig.supxlabel(r"cell values $\times 10^{-3}$", fontsize=6)

    return fig, "shapley_interaction_heatmap"


def fig_6_6c(df):
    """Plot ladder Shapley values (MNLI and SQuAD) as grouped bars.

    Contributors: rank, subset_size, learning_rate and training_regime;
    lora_target is always "attention" in the ladder.
    """
    shapley_df, _ = all_ladder_attributions(df)

    ladder_labels = {d: MAIN_DATASET_LABELS[d] for d in LADDER_DATASETS}

    fig = _grouped_shapley_bars(
        shapley_df, list(LADDER_CONTRIBUTORS.keys()), ladder_labels,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * (0.45 / 0.6)),
    )

    return fig, "shapley_ladder"


def fig_6_6d(df):
    """Plot Shapley values of rank and scaling ratio as grouped bars.

    All five ranks, each run at both ratios, so rank and ratio vary
    independently; lora_target and learning_rate are held at the
    defaults (qv, 3e-4). SQuAD's rank bar is exactly zero because the
    default rank 8 is already the best rank in this slice.
    """
    shapley_df, _ = all_rank_ratio_attributions(df)

    fig = _grouped_shapley_bars(
        shapley_df, list(RANK_RATIO_CONTRIBUTORS.keys()), MAIN_DATASET_LABELS,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.45),
        annotate_exact_zero=True,
    )

    return fig, "shapley_rank_vs_ratio"


# HyperSHAP figures (Section 6.6), read from the CSV exported by
# scripts/export_hypershap_values.py; this module never imports
# hypershap itself. They are fig_6_6e/6_6f and do not replace the exact
# Shapley figures fig_6_6a-d.

HYPERSHAP_VALUES_CSV = "results/hypershap_values.csv"

# Fixed x-axis order for both HyperSHAP figures below.
HYPERSHAP_PLAYERS = ["learning_rate", "lora_target", "rank", "scaling_ratio"]


def _load_hypershap_values(section):
    """Return the rows of one `section` of the HyperSHAP values CSV."""
    hs = pd.read_csv(HYPERSHAP_VALUES_CSV)

    return hs[hs["section"] == section]


def _pretty_players(players):
    """Return a readable label for one player or a "+"-joined pair.

    Names go through pretty(); pairs are joined with a mathtext times
    sign, since cmr10 has no glyph for the Unicode one.
    """
    return " $\\times$ ".join(pretty(p) for p in players.split("+"))


# Short labels for fig_6_6f's crowded x-axis only (10 categories per
# panel); everywhere else the full pretty() names are used.
HYPERSHAP_SHORT_LABELS = {
    "learning_rate": "LR",
    "lora_target": "Target",
    "rank": "r",
    "scaling_ratio": r"$\alpha$/$r$",
}


def _short_players(players):
    """Return the short label of _pretty_players() for fig_6_6f."""
    return " $\\times$ ".join(HYPERSHAP_SHORT_LABELS[p] for p in players.split("+"))


def _hypershap_stack_components(sub_dataset, player):
    """Return (order1, order2_pos, order2_neg) of one hyperparameter.

    order1 is its own order-1 value; order2_pos and order2_neg sum its
    positive and negative order-2 terms. Used by fig_6_6e, where each
    pairwise term counts in both members' bars.
    """
    order1 = sub_dataset.loc[
        (sub_dataset["order"] == 1) & (sub_dataset["players"] == player), "value"
    ].iloc[0]

    o2 = sub_dataset[
        (sub_dataset["order"] == 2)
        & sub_dataset["players"].apply(lambda p: player in p.split("+"))
    ]

    order2_pos = o2.loc[o2["value"] > 0, "value"].sum()
    order2_neg = o2.loc[o2["value"] < 0, "value"].sum()

    return order1, order2_pos, order2_neg


def fig_6_6e(df):
    """Plot HyperSHAP main-sweep sensitivity as stacked bars.

    One panel per dataset, one bar per hyperparameter. Each bar stacks
    its order-1 value and the sum of all order-2 terms it is part of;
    positive parts go up, negative parts down. Each pairwise term counts
    in both members' bars, so the bars do not add up.

    Values are in units of 10^-3. MNLI, SQuAD and CoNLL-2003 share a
    y-axis; GSM8K's values are about 250x smaller, so it gets its own
    axis (noted in the panel) instead of looking as large as the others.
    """
    sens = _load_hypershap_values("main_sweep_sensitivity")

    order1_color, order2_color = CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[1]

    # Verification against the known values (tolerance 1e-6), run on
    # every build so that a changed CSV fails loudly.

    mnli = sens[sens["dataset"] == "mnli"]

    _, lr_pos, _ = _hypershap_stack_components(mnli, "learning_rate")
    _, rank_pos, _ = _hypershap_stack_components(mnli, "rank")

    assert abs(lr_pos - 0.041897) < 1e-6, f"MNLI learning_rate order-2 positive stack: {lr_pos}"
    assert abs(rank_pos - 0.041896) < 1e-6, f"MNLI rank order-2 positive stack: {rank_pos}"

    mnli_o2 = mnli[mnli["order"] == 2]
    assert (mnli_o2["value"] > 0).all(), (
        f"MNLI: expected every order-2 value positive, got "
        f"{mnli_o2[['players', 'value']].to_dict('records')}"
    )

    for dataset in ["mnli", "squad", "conll2003"]:

        o1 = sens[(sens["dataset"] == dataset) & (sens["order"] == 1)]

        assert (o1["value"] < 0).all(), (
            f"{dataset}: expected every order-1 value negative, got "
            f"{o1[['players', 'value']].to_dict('records')}"
        )

    # --- Figure ---

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.9),
        sharey=False,
    )

    x_positions = list(range(len(HYPERSHAP_PLAYERS)))
    shared_axes = []

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        sub = sens[sens["dataset"] == dataset]
        is_gsm8k = dataset == "gsm8k"
        scale = 1e3  # Units of 10^-3.

        for x, player in zip(x_positions, HYPERSHAP_PLAYERS):

            o1, o2_pos, o2_neg = _hypershap_stack_components(sub, player)
            o1, o2_pos, o2_neg = o1 * scale, o2_pos * scale, o2_neg * scale

            up = ([("order1", o1)] if o1 > 0 else []) + ([("order2", o2_pos)] if o2_pos > 0 else [])
            down = ([("order1", o1)] if o1 < 0 else []) + ([("order2", o2_neg)] if o2_neg < 0 else [])

            bottom = 0
            for label, height in up:
                ax.bar(
                    x, height, bottom=bottom, width=0.6, zorder=2,
                    color=order1_color if label == "order1" else order2_color,
                )
                bottom += height

            bottom = 0
            for label, height in down:
                ax.bar(
                    x, height, bottom=bottom, width=0.6, zorder=2,
                    color=order1_color if label == "order1" else order2_color,
                )
                bottom += height

        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([pretty(p) for p in HYPERSHAP_PLAYERS], rotation=30, ha="right")
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel("HyperSHAP attribution ($\\times10^{-3}$)")

        if is_gsm8k:
            ax.text(
                0.97, 0.97, "note: own axis scale",
                ha="right", va="top", fontsize=6.5, transform=ax.transAxes,
            )
        else:
            shared_axes.append(ax)

    # Shared y-limits for MNLI, SQuAD and CoNLL-2003 only, taken from
    # their autoscaled ranges after drawing.
    shared_lo = min(ax.get_ylim()[0] for ax in shared_axes)
    shared_hi = max(ax.get_ylim()[1] for ax in shared_axes)
    for ax in shared_axes:
        ax.set_ylim(shared_lo, shared_hi)

    legend_handles = [
        Patch(facecolor=order1_color, label="Order 1"),
        Patch(facecolor=order2_color, label="Order 2"),
    ]
    fig.legend(handles=legend_handles, loc="outside upper center", ncol=2, frameon=False)

    return fig, "hypershap_main_sweep"


def fig_6_6f(df):
    """Plot the HyperSHAP safe-to-risky ablation as waterfalls.

    Ten contributions (4 order-1, 6 order-2) explain the score gap
    between the safe (qv, r=4, lr 1e-4) and the risky (full, r=64,
    lr 5e-4) configuration, one panel per dataset, sorted ascending.
    Bar color marks the sign; each panel states the total f(x), which
    equals the measured score gap. Values are in units of 10^-3.

    The x-axis uses short labels (LR, Target, r, alpha/r); state this
    key in the caption. The four scaling_ratio terms are zero by
    construction (both endpoints use ratio 2). They are shown greyed
    and hatched without value labels, and do not mean that the ratio
    is unimportant in general.
    """
    abl = _load_hypershap_values("ablation_safe_to_risky")
    abl = abl[abl["order"] != 0]  # Drop the baseline row.

    positive_color, negative_color = CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[1]
    null_color = "0.85"

    # Measured value(risky) - value(safe) per dataset.
    expected_sums = {"mnli": -0.711752, "squad": -0.843433, "conll2003": -0.931007, "gsm8k": -0.116000}

    # --- Verification assertions ---

    for dataset, expected in expected_sums.items():

        total = abl.loc[abl["dataset"] == dataset, "value"].sum()

        assert abs(total - expected) < 1e-6, (
            f"{dataset}: total f(x) {total:.6f} does not match expected {expected} "
            f"(tolerance 1e-6)"
        )

    for dataset in MAIN_DATASETS:

        d = abl[abl["dataset"] == dataset]
        scaling_ratio_terms = d[d["players"].apply(lambda p: "scaling_ratio" in p.split("+"))]

        assert len(scaling_ratio_terms) == 4, (
            f"{dataset}: expected 4 scaling_ratio-involving terms (1 order-1 + 3 "
            f"order-2 pairs), found {len(scaling_ratio_terms)}"
        )
        assert (scaling_ratio_terms["value"].abs() <= NUMERICAL_ZERO).all(), (
            f"{dataset}: expected every scaling_ratio-involving term within "
            f"{NUMERICAL_ZERO:g} of zero (structurally forced by the ablation's fixed "
            f"scaling_ratio=2 endpoints), got "
            f"{scaling_ratio_terms[['players', 'value']].to_dict('records')}"
        )

    top3_pairs = {"learning_rate+rank", "learning_rate+lora_target", "lora_target+rank"}

    for dataset in ["mnli", "squad", "conll2003"]:

        d = abl[abl["dataset"] == dataset].assign(absval=lambda x: x["value"].abs())
        top3 = set(d.sort_values("absval", ascending=False).head(3)["players"])

        assert top3 == top3_pairs, (
            f"{dataset}: expected the 3 largest-magnitude contributions to be "
            f"{top3_pairs}, got {top3}"
        )

        o1 = d[(d["order"] == 1) & d["players"].isin(["learning_rate", "lora_target", "rank"])]

        assert (o1["value"] > 0).all(), (
            f"{dataset}: expected learning_rate/lora_target/rank order-1 terms all "
            f"positive, got {o1[['players', 'value']].to_dict('records')}"
        )

    # --- Figure ---

    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.95),
        sharex=False, sharey=False,
    )

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        d = abl[abl["dataset"] == dataset].sort_values("value", ascending=True).reset_index(drop=True)
        scale = 1e3  # Units of 10^-3.

        running = 0.0

        for i, row in d.iterrows():

            is_null = "scaling_ratio" in row["players"].split("+")
            color = null_color if is_null else (positive_color if row["value"] > 0 else negative_color)

            ax.bar(
                i, row["value"] * scale, bottom=running, width=0.7, zorder=2,
                color=color, edgecolor="0.3", linewidth=0.6,
                hatch="////" if is_null else None,
            )

            top = running + row["value"] * scale

            # No value label on the null bars: their labels would all be
            # "0" and overlap; hatching and legend mark them.
            if not is_null:

                # Vertical label above the bar's upper end, where a
                # waterfall column is always free; adjacent labels
                # cannot overlap horizontally.
                ax.annotate(
                    fmt_e3(row["value"]), xy=(i, max(running, top)), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, rotation=90,
                )

            running = top

        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(
            [_short_players(p) for p in d["players"]], rotation=90, fontsize=6.5,
        )
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel("HyperSHAP attribution ($\\times10^{-3}$)")

        # Larger top padding: all totals are negative, so the bars touch
        # y=0 and the "Total f(x)" annotation and the first bar's label
        # need room above the line.
        y_bottom, y_top = ax.get_ylim()
        span = y_top - y_bottom
        ax.set_ylim(y_bottom - span * 0.08, y_top + span * 0.30)

        total = d["value"].sum()
        ax.text(
            0.97, 0.95, f"Total $f(x)$ = {fmt_e3(total)}",
            ha="right", va="top", fontsize=6.5, transform=ax.transAxes,
        )

    legend_handles = [
        Patch(facecolor=positive_color, label="Positive"),
        Patch(facecolor=negative_color, label="Negative"),
        Patch(facecolor=null_color, edgecolor="0.3", hatch="////", label="Scaling ratio (structurally 0)"),
    ]
    fig.legend(handles=legend_handles, loc="outside upper center", ncol=3, frameon=False)

    return fig, "hypershap_ablation"


# Chapter 7 discussion figures: post-hoc analyses of existing runs.

BOOTSTRAP_SEED = 0


def _expected_best_under_budget(config_means, k_values, n_draws=1000, rng=None):
    """Return the expected best result for search budgets k.

    For each k, draw k configuration means without replacement, take
    the max and repeat n_draws times. Returns k, the mean of the maxima
    and their 25th/75th percentiles.
    """
    rng = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(config_means)
    n_configs = len(values)

    rows = []

    for k in k_values:

        draws = np.array([
            values[rng.choice(n_configs, size=k, replace=False)].max()
            for _ in range(n_draws)
        ])

        rows.append({
            "k": k,
            "mean_best": draws.mean(),
            "q25": np.percentile(draws, 25),
            "q75": np.percentile(draws, 75),
        })

    return pd.DataFrame(rows)


def fig_7_1a(df):
    """Plot the expected best result under a search budget of k runs.

    Bootstraps from the main sweep without new training: draw k of a
    target's 30 configuration means, take the max, repeat 1000 times.
    One line per target with its interquartile band, one panel per
    dataset. Shows how many runs each target needs to reach its
    ceiling. The zero-shot mean is shown as corner text.
    """
    main = _main_sweep_data(df)

    config_means_all = (
        main.groupby(["dataset", "lora_target", "rank", "alpha", "learning_rate"], as_index=False)["metric"]
        .mean()
    )

    # 2x2 instead of 1x4, for wider panels.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.78),
        sharey=False,
    )

    k_values = list(range(1, MAIN_N_CONFIGS + 1))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    zeroshot_means = _zeroshot_means(df)

    for ax, dataset in zip(axes.flat, MAIN_DATASETS):

        for target in MAIN_TARGETS:

            values = config_means_all.loc[
                (config_means_all["dataset"] == dataset) & (config_means_all["lora_target"] == target),
                "metric",
            ]

            assert len(values) == MAIN_N_CONFIGS, (
                f"expected {MAIN_N_CONFIGS} configurations for ({dataset}, {target}), found "
                f"{len(values)} -- the main sweep grid is assumed complete"
            )

            result = _expected_best_under_budget(values, k_values, rng=rng)
            color = STRATEGY_COLORS[target]

            ax.plot(result["k"], result["mean_best"], color=color, linewidth=1.2, label=pretty(target))
            ax.fill_between(result["k"], result["q25"], result["q75"], color=color, alpha=0.15, linewidth=0)

        ax.text(
            0.97, 0.03, f"Zero-shot mean: {zeroshot_means[dataset]:.3f}",
            ha="right", va="bottom", fontsize=5.5, transform=ax.transAxes,
        )

        ax.set_xlabel("Search budget k")
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(f"Expected best {METRIC_NAMES[dataset].lower()}")

    # axes is 2x2, so take the legend handles from the first Axes.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=len(MAIN_TARGETS),
        frameon=False,
    )

    return fig, "expected_best_under_budget"


def _ci_width_vs_k(seed_values, k_values, n_bootstrap=5000, rng=None):
    """Return the bootstrap 95% CI width of the mean for k seeds.

    Resamples k seeds with replacement n_bootstrap times; the width is
    the 97.5th minus the 2.5th percentile of the means. Unlike a
    formula-based CI this also works for k=1, where it equals the raw
    seed spread.
    """
    rng = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(seed_values)

    rows = []

    for k in k_values:

        boot_means = np.array([
            rng.choice(values, size=k, replace=True).mean()
            for _ in range(n_bootstrap)
        ])

        lower, upper = np.percentile(boot_means, [2.5, 97.5])
        rows.append({"k": k, "ci_width": upper - lower})

    return pd.DataFrame(rows)


def fig_7_3a(df):
    """Plot the 95% CI width of the mean against the number of seeds.

    Bootstraps from the FFT-stability cell with the most seeds and no
    collapsed seed (a collapse would make the spread bimodal; MNLI wins
    ties). The width at k=3 is the uncertainty of a 3-seed cell in the
    main sweep.
    """
    stability = _fft_stability_data(df)
    eligible = pd.concat([stability, df[df["block"] == "zeroshot"]])
    collapsed = is_collapsed(eligible)
    stability = stability.assign(_collapsed=collapsed.loc[stability.index])

    cell_stats = (
        stability.groupby(["dataset", "learning_rate"])
        .agg(n=("seed", "count"), n_collapsed=("_collapsed", "sum"))
        .reset_index()
    )
    healthy = cell_stats[cell_stats["n_collapsed"] == 0]

    assert not healthy.empty, "fig_7_3a: no fully-healthy FFT-stability cell found"

    max_n = healthy["n"].max()
    candidates = healthy[healthy["n"] == max_n]
    chosen = (
        candidates[candidates["dataset"] == "mnli"].iloc[0]
        if (candidates["dataset"] == "mnli").any()
        else candidates.sort_values("dataset").iloc[0]
    )

    seed_values = stability.loc[
        (stability["dataset"] == chosen["dataset"]) & (stability["learning_rate"] == chosen["learning_rate"]),
        "metric",
    ]
    n_max = int(chosen["n"])

    k_values = list(range(1, n_max + 1))
    result = _ci_width_vs_k(seed_values, k_values)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * (0.45 / 0.6)))

    ax.plot(result["k"], result["ci_width"], color=STRATEGY_COLORS["fft"], marker="o", markersize=4)
    ax.axvline(3, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(3.15, ax.get_ylim()[1] * 0.9, "n=3 (main sweep)", fontsize=6)
    ax.text(
        0.97, 0.95, f"{MAIN_DATASET_LABELS[chosen['dataset']]} FFT, {fmt_lr(chosen['learning_rate'])}",
        ha="right", va="top", fontsize=6, transform=ax.transAxes,
    )

    ax.set_xticks(k_values)
    ax.set_xlabel("Number of seeds (k)")
    ax.set_ylabel("95% CI width")

    return fig, "ci_width_vs_seeds"


def fig_7_1b(df):
    """Plot reliability (grid mean) vs. peak (best run) per target.

    One point per (target, dataset), colored by target, with the line
    y = x: points far below it have a high ceiling but an unreliable
    grid. This is the one figure that pools the datasets' different
    metrics on one pair of axes (labeled "own scale"; state this in the
    caption). Labels are placed by adjustText; the axes are square.
    """
    main = _main_sweep_data(df)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN))

    all_values = []
    texts = []

    for dataset in MAIN_DATASETS:
        for target in MAIN_TARGETS:

            cell = main[(main["dataset"] == dataset) & (main["lora_target"] == target)]
            config_means = cell.groupby(["rank", "alpha", "learning_rate"])["metric"].mean()

            grid_mean = config_means.mean()
            grid_best = config_means.max()
            all_values.extend([grid_mean, grid_best])

            ax.scatter(
                grid_mean, grid_best, color=STRATEGY_COLORS[target],
                s=30, zorder=3, edgecolors="black", linewidths=0.4,
            )

            texts.append(ax.text(
                grid_mean, grid_best, MAIN_DATASET_LABELS[dataset],
                fontsize=5.5, zorder=4,
            ))

    lo, hi = min(all_values) - 0.02, max(all_values) + 0.02
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    adjust_text(
        texts, ax=ax,
        force_text=(0.6, 0.8), expand=(1.3, 1.6),
        arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.5),
    )

    ax.set_xlabel("Grid mean (each dataset's own scale)")
    ax.set_ylabel("Grid best (each dataset's own scale)")

    legend_handles = [
        Line2D([0], [0], marker="o", color=STRATEGY_COLORS[t], linestyle="", markersize=6, label=pretty(t))
        for t in MAIN_TARGETS
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6, loc="lower right")

    return fig, "reliability_vs_peak"


def fig_7_2a(df):
    """Plot the best learning rate per strategy and dataset (log scale).

    For each LoRA target the rate with the best mean over all (rank,
    alpha) pairs, for FFT the best of its three rates. The band spans
    FFT's best rate to 10x that rate. With only three disjoint
    candidates per strategy this can show consistency with a ~10x
    factor, not estimate it (state this in the caption). Strategies are
    offset horizontally so that equal points stay visible.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    core = core[core["adaptation"] != "none"]
    core = attach_strategy(core)

    strategies = ["fft", "qv", "attention", "full"]
    dodge = {"fft": -0.24, "qv": -0.08, "attention": 0.08, "full": 0.24}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * (0.55 / 0.75)))

    for x_idx, dataset in enumerate(MAIN_DATASETS):

        sub = core[core["dataset"] == dataset]
        fft_best_lr = sub.loc[sub["strategy"] == "fft"].groupby("learning_rate")["metric"].mean().idxmax()

        ax.axvspan(
            x_idx - 0.4, x_idx + 0.4, ymin=0, ymax=1, color="none",
        )
        ax.fill_betweenx(
            [fft_best_lr, fft_best_lr * 10], x_idx - 0.35, x_idx + 0.35,
            color=STRATEGY_COLORS["fft"], alpha=0.1, linewidth=0, zorder=0,
        )

        for strategy in strategies:

            strategy_lr_means = sub.loc[sub["strategy"] == strategy].groupby("learning_rate")["metric"].mean()
            best_lr = strategy_lr_means.idxmax()

            ax.scatter(
                x_idx + dodge[strategy], best_lr, color=STRATEGY_COLORS[strategy],
                s=40, zorder=3, edgecolors="black", linewidths=0.4,
            )

    # A real log axis here, since the ~10x band needs true magnitudes;
    # the ticks are labeled via fmt_lr.
    all_lrs = sorted(set(FFT_LEARNING_RATES) | set(MAIN_LEARNING_RATES))
    ax.set_yscale("log")
    ax.set_yticks(all_lrs)
    ax.set_yticklabels([fmt_lr(lr) for lr in all_lrs], fontsize=6)
    ax.minorticks_off()
    ax.set_xticks(range(len(MAIN_DATASETS)))
    ax.set_xticklabels([MAIN_DATASET_LABELS[d] for d in MAIN_DATASETS])
    ax.set_ylabel("Best learning rate")

    legend_handles = [
        Line2D([0], [0], marker="o", color=STRATEGY_COLORS[s], linestyle="", markersize=6, label=pretty(s))
        for s in strategies
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6, loc="upper left")

    return fig, "optimal_lr_lora_vs_fft"


def fig_7_2b(df):
    """Plot whether rank matters, at a fixed and at the best LR.

    qv only, one panel per dataset. Line (i) uses a fixed lr 3e-4, line
    (ii) the best learning rate per rank; both average over alpha. If
    (ii) is flat where (i) is not, the rank effect comes from not
    re-tuning the learning rate (Lee et al.).
    """
    main = _main_sweep_data(df)
    main = main[main["lora_target"] == "qv"]

    fig, axes = plt.subplots(
        1, len(MAIN_DATASETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.34),
        sharex=True,
    )

    fixed_lr = 3e-4

    for ax, dataset in zip(axes, MAIN_DATASETS):

        sub = main[main["dataset"] == dataset]

        # Evenly spaced categories; on a log axis the rank labels would
        # run together.
        rank_positions = categorical_axis(ax, RANK_VALUES, str)

        per_rank_lr = sub.groupby(["rank", "learning_rate"])["metric"].mean().reset_index()

        fixed_curve = (
            per_rank_lr[per_rank_lr["learning_rate"] == fixed_lr]
            .set_index("rank")["metric"]
            .sort_index()
        )
        best_curve = per_rank_lr.groupby("rank")["metric"].max().sort_index()

        # Reuse the two LADDER_LR_COLORS for the two lines.
        ax.plot(
            [rank_positions[r] for r in fixed_curve.index], fixed_curve.values,
            marker="o", markersize=3, label=f"Fixed LR ({fmt_lr(fixed_lr)})", color=LADDER_LR_COLORS[1e-4],
        )
        ax.plot(
            [rank_positions[r] for r in best_curve.index], best_curve.values,
            marker="o", markersize=3, label="Per-rank best LR", color=LADDER_LR_COLORS[3e-4],
        )

        ax.set_xlabel(pretty("rank"))
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(f"{METRIC_NAMES[dataset]} (qv)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=2,
        frameon=False,
    )

    return fig, "does_rank_matter"


FIXED_LR_REPAIRED = 3e-4


def _fig_7_2b_repaired_stats(df, n_boot=10000, n_null_sim=200000, boot_seed=0, null_seed=1):
    """Return the numbers plotted by fig_7_2b_v2 (qv, ratio 2 only).

    Shared with the reporting script, so the numbers in the text match
    the figure. Returns {dataset: {"fixed_mean", "fixed_sd",
    "best_mean", "best_lo", "best_hi", "best_lr", "sigma",
    "selection_bias"}}, with arrays in RANK_VALUES order.

    fixed_*: the lr 3e-4 cell per rank. best_*: the best of the three
    learning rates, with a seed bootstrap (n_boot) that re-runs the
    selection in every replicate. sigma is the median seed SD of the
    dataset's 15 cells, selection_bias the simulated expected max of
    three noisy 3-seed means under no real effect.
    """
    rng_boot = np.random.default_rng(boot_seed)
    rng_null = np.random.default_rng(null_seed)

    main = _main_sweep_data(df)
    main = main[(main["lora_target"] == "qv") & (main["scaling_ratio"] == 2.0)]

    results = {}

    for dataset in MAIN_DATASETS:

        sub = main[main["dataset"] == dataset]

        cells = {}
        for rank in RANK_VALUES:
            for lr in MAIN_LEARNING_RATES:
                vals = sub.loc[(sub["rank"] == rank) & (sub["learning_rate"] == lr), "metric"].values
                assert len(vals) == 3, f"{dataset} r={rank} lr={lr}: expected 3 seeds, found {len(vals)}"
                cells[(rank, lr)] = vals

        seed_sds = [cells[(r, lr)].std(ddof=1) for r in RANK_VALUES for lr in MAIN_LEARNING_RATES]
        sigma = float(np.median(seed_sds))
        se_of_mean = sigma / np.sqrt(3)

        null_draws = rng_null.normal(0, se_of_mean, size=(n_null_sim, 3))
        selection_bias = float(null_draws.max(axis=1).mean())

        fixed_mean, fixed_sd = [], []
        best_mean, best_lo, best_hi, best_lr = [], [], [], []

        for rank in RANK_VALUES:

            fv = cells[(rank, FIXED_LR_REPAIRED)]
            fixed_mean.append(fv.mean())
            fixed_sd.append(fv.std(ddof=1))

            lr_means = {lr: cells[(rank, lr)].mean() for lr in MAIN_LEARNING_RATES}
            argmax_lr = max(lr_means, key=lr_means.get)
            best_mean.append(lr_means[argmax_lr])
            best_lr.append(argmax_lr)

            boot_maxes = np.empty(n_boot)
            for b in range(n_boot):
                resampled = [
                    rng_boot.choice(cells[(rank, lr)], size=3, replace=True).mean()
                    for lr in MAIN_LEARNING_RATES
                ]
                boot_maxes[b] = max(resampled)
            lo, hi = np.percentile(boot_maxes, [2.5, 97.5])
            best_lo.append(lo)
            best_hi.append(hi)

        results[dataset] = {
            "fixed_mean": np.array(fixed_mean), "fixed_sd": np.array(fixed_sd),
            "best_mean": np.array(best_mean), "best_lo": np.array(best_lo),
            "best_hi": np.array(best_hi), "best_lr": best_lr,
            "sigma": sigma, "selection_bias": selection_bias,
        }

    return results


def fig_7_2b_v2(df):
    """Repaired fig_7_2b, kept separately for comparison (candidate).

    Same question and layout as fig_7_2b, with three fixes:

    1. Only scaling ratio 2, so alpha is no longer mixed in.
    2. Error bars: the seed SD at lr 3e-4, and for the best-LR line a
       bootstrap that re-runs the LR selection in every replicate.
    3. A shaded band for the expected lift from picking the best of
       three noisy means (selection bias); points inside it are no
       evidence of a real per-rank LR effect.

    Re-tuning the LR shrinks the rank spread on SQuAD, CoNLL-2003 and
    slightly on MNLI, but widens it on GSM8K (which is noisier), so the
    caption states only what each panel shows.
    """
    stats = _fig_7_2b_repaired_stats(df)

    fig, axes = plt.subplots(
        1, len(MAIN_DATASETS),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42),
        sharex=True,
    )

    for ax, dataset in zip(axes, MAIN_DATASETS):

        s = stats[dataset]
        rank_positions = categorical_axis(ax, RANK_VALUES, str)
        x = np.array([rank_positions[r] for r in RANK_VALUES])

        null_top = s["fixed_mean"] + s["selection_bias"]

        ax.fill_between(
            x, s["fixed_mean"], null_top, color="0.5", alpha=0.18, linewidth=0,
            zorder=0, label="Selection-alone null band",
        )

        ax.errorbar(
            x, s["fixed_mean"], yerr=s["fixed_sd"], marker="o", markersize=3,
            label=f"Fixed LR ({fmt_lr(FIXED_LR_REPAIRED)})", color=LADDER_LR_COLORS[1e-4],
            capsize=2, linewidth=1.2, elinewidth=0.8,
        )
        ax.errorbar(
            x, s["best_mean"],
            yerr=[s["best_mean"] - s["best_lo"], s["best_hi"] - s["best_mean"]],
            marker="o", markersize=3, label="Per-rank best LR",
            color=LADDER_LR_COLORS[3e-4], capsize=2, linewidth=1.2, elinewidth=0.8,
        )

        ax.set_xlabel(pretty("rank"))
        ax.set_title(MAIN_DATASET_LABELS[dataset])
        ax.set_ylabel(f"{METRIC_NAMES[dataset]} (qv, ratio=2)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=3,
        frameon=False, fontsize=7,
    )

    return fig, "does_rank_matter_repaired"


def fig_7_3b(df):
    """Plot the within-cell seed SD by dataset (evaluation path noise).

    One point per cell and a bar at each dataset's median. MNLI is
    scored without generation, so its SD is training noise only; the
    generative datasets add generation nondeterminism. MNLI is thus a
    lower bound, not a zero-noise baseline. Collapsed runs are excluded
    first (is_collapsed()); cells with fewer than two remaining seeds
    are dropped. Uses CORE_BLOCKS.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]
    collapsed = is_collapsed(core)
    healthy = core[~collapsed]

    per_cell = (
        healthy.groupby(CELL_COLS, dropna=False)["metric"]
        .agg(std="std", n="count")
        .reset_index()
    )
    per_cell = per_cell[per_cell["n"] >= 2]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * (0.55 / 0.7)))

    # "control" vs. "generative" is not a strategy, so it uses the
    # categorical palette.
    noise_colors = categorical_color(["control", "generative"])

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for x_idx, dataset in enumerate(MAIN_DATASETS):

        values = per_cell.loc[per_cell["dataset"] == dataset, "std"].dropna()

        color = noise_colors["control"] if dataset == "mnli" else noise_colors["generative"]

        ax.bar(x_idx, values.median(), color=color, alpha=0.25, width=0.6, zorder=1)

        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(x_idx + jitter, values, color=color, s=6, alpha=0.5, linewidths=0, zorder=2)

    ax.set_xticks(range(len(MAIN_DATASETS)))
    labels = [MAIN_DATASET_LABELS[d] for d in MAIN_DATASETS]
    labels[MAIN_DATASETS.index("mnli")] += "\n(control, deterministic)"
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Within-cell SD (bar = median)")

    return fig, "within_cell_noise"


# Appendix figures (docs/plot_plan.md §10).

def _fig_A1_color_bounds(df, dataset):
    """Return (p5, vmax) shared by the three target panels of a dataset.

    Pooled over all targets of `dataset`, so equal colors mean equal
    values across its panels. Not pooled across datasets, since their
    metrics differ.
    """
    main = _main_sweep_data(df)
    cell = main[main["dataset"] == dataset]

    values = []
    for target in MAIN_TARGETS:
        for rank, alpha in MAIN_RANK_ALPHA_PAIRS:
            for lr in MAIN_LEARNING_RATES:
                match = cell[
                    (cell["lora_target"] == target) & (cell["rank"] == rank)
                    & (cell["alpha"] == alpha) & (cell["learning_rate"] == lr)
                ]
                values.append(match["metric"].mean())

    values = np.array(values, dtype=float)

    return np.nanpercentile(values, 5), np.nanmax(values)


def _fig_A1_single(df, dataset, target):
    """Plot one A1 heatmap of the mean metric per LR and (r, alpha).

    One figure per (dataset, target); learning rates on the y-axis, the
    ten (rank, alpha) pairs on the x-axis. The color scale is shared
    by the dataset's three target panels (_fig_A1_color_bounds).

    RdYlGn (green = good) was requested, although it is not
    colorblind-safe; every cell also shows its value as text. vmin is
    the pooled 5th percentile, so collapsed outliers do not wash out
    the healthy range; cells below it are hatched and the colorbar gets
    a "min" extension.
    """
    main = _main_sweep_data(df)
    cell = main[(main["dataset"] == dataset) & (main["lora_target"] == target)]

    n_rows, n_cols = len(MAIN_LEARNING_RATES), len(MAIN_RANK_ALPHA_PAIRS)
    matrix = np.full((n_rows, n_cols), np.nan)

    for i, lr in enumerate(MAIN_LEARNING_RATES):
        for j, (rank, alpha) in enumerate(MAIN_RANK_ALPHA_PAIRS):

            match = cell[(cell["rank"] == rank) & (cell["alpha"] == alpha) & (cell["learning_rate"] == lr)]
            matrix[i, j] = match["metric"].mean()

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42))

    cmap = plt.get_cmap("RdYlGn")
    p5, vmax = _fig_A1_color_bounds(df, dataset)
    norm = plt.Normalize(vmin=p5, vmax=vmax)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, label=f"Mean {METRIC_NAMES[dataset].lower()}", extend="min")

    for i in range(n_rows):
        for j in range(n_cols):

            text_color = annot_color(matrix[i, j], cmap, norm)
            # Outline the digits in the opposite color, since the hatch
            # uses the text color and would cross through them.
            stroke_color = "black" if text_color == "white" else "white"

            ax.text(
                j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                color=text_color, fontsize=7,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground=stroke_color)],
            )

            if matrix[i, j] < p5:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="////", edgecolor=text_color, linewidth=0,
                ))

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([fmt_lr(lr) for lr in MAIN_LEARNING_RATES])
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [rf"$r$={r}, $\alpha$={a}" for r, a in MAIN_RANK_ALPHA_PAIRS],
        rotation=45, ha="right",
    )
    ax.set_ylabel(pretty("learning_rate"))
    ax.set_xlabel(pretty("rank_alpha"))

    return fig, f"mean_heatmap_{dataset}_{target}"


def fig_A2(df):
    """Plot the sub-metrics per strategy (MNLI, CoNLL-2003, SQuAD).

    Precision, recall and F1 (plus accuracy for MNLI; exact match and
    F1 for SQuAD), read from the result JSONs via raw_metrics.py since
    the CSV lacks them. GSM8K has no sub-metrics and is left out. One
    shared legend outside the axes, so that no bars are hidden.
    """
    strategies = ["zeroshot", "fft", "qv", "attention", "full"]
    block_for_strategy = {
        "zeroshot": "zeroshot", "fft": "fft", "qv": "main_qv",
        "attention": "main_attention", "full": "main_full",
    }
    datasets = ["mnli", "conll2003", "squad"]
    # Sub-metrics use the categorical palette, not the strategy colors.
    all_keys = sorted(set().union(*[SUB_METRIC_KEYS[d] for d in datasets]))
    submetric_colors = categorical_color(all_keys)

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.48),
    )

    for ax, dataset in zip(axes, datasets):

        sub_df = load_sub_metrics(dataset, list(block_for_strategy.values()))
        keys = SUB_METRIC_KEYS[dataset]
        width = 0.8 / len(keys)

        for k_idx, key in enumerate(keys):

            offsets = [
                s_idx + (k_idx - (len(keys) - 1) / 2) * width
                for s_idx in range(len(strategies))
            ]
            values = [
                sub_df.loc[sub_df["block"] == block_for_strategy[s], key].mean()
                for s in strategies
            ]

            ax.bar(
                offsets, values, width=width, color=submetric_colors[key],
                label=pretty(key), edgecolor="black", linewidth=0.3,
            )

        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([pretty(s) for s in strategies], rotation=45, ha="right", fontsize=6.5)
        ax.set_title(MAIN_DATASET_LABELS[dataset])

    axes[0].set_ylabel("Score")

    legend_handles = [
        Patch(facecolor=submetric_colors[key], edgecolor="black", linewidth=0.3, label=pretty(key))
        for key in all_keys
    ]
    fig.legend(
        handles=legend_handles, loc="outside upper center", ncol=len(all_keys),
        frameon=False,
    )

    return fig, "sub_metrics"


def _match_loss_file(metrics_path):
    """Return the loss-curve file written closest in time to a result.

    Result and loss files of one run get separate unique suffixes, so
    they are matched by modification time, not by name.
    """
    run_dir = os.path.dirname(os.path.dirname(metrics_path))
    loss_dir = os.path.join(run_dir, "loss_curves")

    candidates = glob.glob(os.path.join(loss_dir, "*.json"))
    assert candidates, f"no loss curve files found in {loss_dir}"

    metrics_mtime = os.path.getmtime(metrics_path)

    return min(candidates, key=lambda p: abs(os.path.getmtime(p) - metrics_mtime))


def _fig_A3_single(df, dataset):
    """Plot training loss curves for one dataset, one panel per block.

    Panels for fft, qv, attention and full (zero-shot never trains), one
    line per seed, at a reference configuration: FFT at lr 1e-5, LoRA at
    lr 3e-4 with (rank, alpha) = (8, 16), the Shapley default. Loss
    files are matched to their runs via _match_loss_file().
    """
    data = df[df["dataset"] == dataset]

    panels = [
        ("fft", 1e-5, None),
        ("main_qv", 3e-4, (8, 16)),
        ("main_attention", 3e-4, (8, 16)),
        ("main_full", 3e-4, (8, 16)),
    ]
    block_to_strategy = {
        "fft": "fft", "main_qv": "qv",
        "main_attention": "attention", "main_full": "full",
    }

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.34),
        sharey=False,
    )

    for ax, (block, lr, rank_alpha) in zip(axes, panels):

        cell = data[(data["block"] == block) & (data["learning_rate"] == lr)]

        if rank_alpha is not None:
            cell = cell[(cell["rank"] == rank_alpha[0]) & (cell["alpha"] == rank_alpha[1])]

        assert len(cell) == 3, (
            f"{dataset}: expected 3 seeds for {block} at lr={lr}, found {len(cell)}"
        )

        # Sort by seed, so the legend order is the same in every panel.
        cell = cell.sort_values("seed")

        for _, row in cell.iterrows():

            loss_path = _match_loss_file(row["source_file"])

            with open(loss_path) as f:
                history = json.load(f)

            # Keep the per-step entries; the final summary record of
            # log_history has no "loss" key.
            step_entries = [entry for entry in history if "loss" in entry]
            steps = [entry["step"] for entry in step_entries]
            losses = [entry["loss"] for entry in step_entries]

            ax.plot(steps, losses, linewidth=0.8, alpha=0.85, label=f"seed {int(row['seed'])}")

        ax.set_xlabel("Step")
        ax.set_title(pretty(block_to_strategy[block]))
        ax.legend(frameon=False, fontsize=5.5)

    axes[0].set_ylabel("Train loss")

    return fig, f"loss_curves_{dataset}"


def fig_A3(df):
    """Plot the MNLI loss curves under the original fig_A3 id."""
    return _fig_A3_single(df, "mnli")


def fig_A5(df):
    """Plot the design coverage: run counts per block and dataset.

    Uses every block in the results, so both the complete core design
    and the gaps of unfinished diagnostic rounds are visible. Block
    names get their own labels (BLOCK_LABELS); cells without runs are
    left blank so that partial counts stand out.
    """
    BLOCK_LABELS = {
        "fft": "FFT",
        "fft_lr_extend": "FFT (LR extend)",
        "fft_wd0.1": "FFT (wd=0.1)",
        "fft_wd0.1_epochs6": "FFT (wd=0.1, 6 epochs)",
        "fft_wd0.1_moreseeds": "FFT (wd=0.1, more seeds)",
        "fft_wd0.1_subset12000": f"FFT (wd=0.1, {fmt_subset(12000)} subset)",
        "fft_wd0.1_subset20000": f"FFT (wd=0.1, {fmt_subset(20000)} subset)",
        "fft_wd0.1_subset7000": f"FFT (wd=0.1, {fmt_subset(7000)} subset)",
        "ladder_isoepoch": "Ladder (iso-epoch)",
        "ladder_isoepoch_full": "Ladder (iso-epoch, full)",
        "ladder_isoepoch_r64_lr5e4": f"Ladder (iso-epoch, r=64, {fmt_lr(5e-4)})",
        "ladder_isoepoch_full_r64_lr5e4": f"Ladder (iso-epoch, full, r=64, {fmt_lr(5e-4)})",
        "ladder_isostep": "Ladder (iso-step)",
        "ladder_isostep_full": "Ladder (iso-step, full)",
        "ladder_isostep_r64_lr5e4": f"Ladder (iso-step, r=64, {fmt_lr(5e-4)})",
        "ladder_isostep_full_r64_lr5e4": f"Ladder (iso-step, full, r=64, {fmt_lr(5e-4)})",
        "main_attention": "Main (attention)",
        "main_attention_riskyseeds": "Main (attention, risky seeds)",
        "main_attention_riskyseeds_lr3e4": f"Main (attention, risky seeds, {fmt_lr(3e-4)})",
        "main_attention_riskyseeds_lr5e4": f"Main (attention, risky seeds, {fmt_lr(5e-4)})",
        "main_full": "Main (full)",
        "main_full_riskyseeds": "Main (full, risky seeds)",
        "main_full_riskyseeds_lr3e4": f"Main (full, risky seeds, {fmt_lr(3e-4)})",
        "main_full_riskyseeds_lr5e4": f"Main (full, risky seeds, {fmt_lr(5e-4)})",
        "main_qv": "Main (qv)",
        "zeroshot": "Zero-shot",
    }

    counts = df.pivot_table(index="block", columns="dataset", values="seed", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=MAIN_DATASETS)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * (0.95 / 0.8)))

    cmap = plt.get_cmap("viridis")
    raw_matrix = counts.values.astype(float)
    matrix = np.where(raw_matrix > 0, raw_matrix, np.nan)
    norm = plt.Normalize(vmin=0, vmax=np.nanmax(matrix))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, label="Run count")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            if raw_matrix[i, j] == 0:
                continue

            text_color = annot_color(matrix[i, j], cmap, norm)

            ax.text(
                j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                color=text_color, fontsize=6.5,
            )

    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels([MAIN_DATASET_LABELS[d] for d in counts.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels([BLOCK_LABELS.get(b, pretty(b)) for b in counts.index], fontsize=6.5)

    return fig, "design_coverage"


# Registry: figure_id -> function. The only place to wire in a new
# figure.

FIGURES = {
    "fig_smoke": fig_smoke,
    "fig_5_1": fig_5_1,
    "fig_6_3a": fig_6_3a,
    "fig_6_3a_lr3e4": fig_6_3a_lr3e4,
    "fig_6_3c": fig_6_3c,
    "fig_6_3d": fig_6_3d,
    "fig_6_3e": fig_6_3e,
    "fig_6_5b": fig_6_5b,
    "fig_6_1a": fig_6_1a,
    "fig_6_1e": fig_6_1e,
    "fig_6_1b": fig_6_1b,
    "fig_6_1c": fig_6_1c,
    "fig_6_2a": fig_6_2a,
    "fig_6_2b": fig_6_2b,
    "fig_6_2c": fig_6_2c,
    "fig_6_4": fig_6_4,
    "fig_6_4d": fig_6_4d,
    "fig_6_4e": fig_6_4e,
    "fig_6_4f": fig_6_4f,
    "fig_6_6a": fig_6_6a,
    "fig_6_6b": fig_6_6b,
    "fig_6_6c": fig_6_6c,
    "fig_6_6d": fig_6_6d,
    "fig_6_6e": fig_6_6e,
    "fig_6_6f": fig_6_6f,
    "fig_7_1a": fig_7_1a,
    "fig_7_1b": fig_7_1b,
    "fig_7_2a": fig_7_2a,
    "fig_7_2b": fig_7_2b,
    "fig_7_2b_v2": fig_7_2b_v2,
    "fig_7_3a": fig_7_3a,
    "fig_7_3b": fig_7_3b,
    "fig_A2": fig_A2,
    "fig_A3": fig_A3,
    "fig_A5": fig_A5,
}

# A1: one figure per (dataset, target).
for _dataset in MAIN_DATASETS:
    for _target in MAIN_TARGETS:
        FIGURES[f"fig_A1_{_dataset}_{_target}"] = functools.partial(
            _fig_A1_single, dataset=_dataset, target=_target
        )
del _dataset, _target

# 6_5a: one figure per dataset.
for _dataset in MAIN_DATASETS:
    FIGURES[f"fig_6_5a_{_dataset}"] = functools.partial(
        _fig_6_5a_single, dataset=_dataset
    )
del _dataset

# A3 for the other datasets; MNLI keeps the original fig_A3 id above.
for _dataset in MAIN_DATASETS:
    if _dataset == "mnli":
        continue
    FIGURES[f"fig_A3_{_dataset}"] = functools.partial(
        _fig_A3_single, dataset=_dataset
    )
del _dataset


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--figure",
        choices=sorted(FIGURES),
        help="Build one registered figure by ID.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Build every registered figure.",
    )

    args = parser.parse_args()

    apply_style()

    df = load_runs()

    figure_ids = sorted(FIGURES) if args.all else [args.figure]

    for figure_id in figure_ids:

        fig, slug = FIGURES[figure_id](df)

        pdf_path, png_path = save_figure(fig, figure_id, slug)

        plt.close(fig)

        print(f"{figure_id}: wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
