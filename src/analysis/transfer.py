"""Configuration transfer and regret analysis (RQ2).

Uses the main sweep only (subset_size=5000): 90 configurations
(3 targets x 10 (rank, alpha) pairs x 3 learning rates), identical on
all four datasets. A configuration can therefore be transferred
between datasets without interpolation; the functions assert that it
exists instead of estimating a value.
"""

import pandas as pd


# Duplicated from plots.py on purpose: plots.py imports this module,
# so importing it here would be circular.
MAIN_SWEEP_BLOCKS = ["main_qv", "main_attention", "main_full"]
CONFIG_COLS = ["lora_target", "rank", "alpha", "learning_rate"]


def _main_sweep_5000(df):

    sweep = df[df["block"].isin(MAIN_SWEEP_BLOCKS)]

    assert (sweep["subset_size"] == 5000).all(), (
        "main sweep blocks are expected to be entirely at subset_size=5000 "
        "(verified true for the current results tree) -- found a different "
        "value, investigate before trusting any transfer result"
    )

    return sweep


def _config_means(df, dataset):
    """Return the seed-mean metric per configuration on `dataset`."""
    sweep = _main_sweep_5000(df)
    sub = sweep[sweep["dataset"] == dataset]

    return (
        sub.groupby(CONFIG_COLS, as_index=False)["metric"]
        .mean()
        .rename(columns={"metric": "mean_metric"})
    )


def _lookup(means, config, match_cols):

    mask = pd.Series(True, index=means.index)

    for col in match_cols:
        mask &= means[col] == config[col]

    return means[mask]


def best_config(df, dataset):
    """Return the configuration with the best seed-mean metric.

    A one-row Series: the four CONFIG_COLS plus mean_metric.
    """
    means = _config_means(df, dataset)

    return means.loc[means["mean_metric"].idxmax()]


def winner_seed_sd(df, dataset):
    """Return the seed SD of `dataset`'s winning configuration.

    Regrets smaller than this cannot be told apart from seed noise.
    """
    winner = best_config(df, dataset)
    sweep = _main_sweep_5000(df)

    match = sweep[
        (sweep["dataset"] == dataset)
        & (sweep["lora_target"] == winner["lora_target"])
        & (sweep["rank"] == winner["rank"])
        & (sweep["alpha"] == winner["alpha"])
        & (sweep["learning_rate"] == winner["learning_rate"])
    ]

    assert len(match) == 3, (
        f"expected exactly 3 seeds for {dataset!r}'s winning config, found "
        f"{len(match)} -- investigate before trusting this SD"
    )

    return match["metric"].std()


def grid_median_regret(df, dataset):
    """Return the median regret of all configurations on `dataset`.

    The cost of a typical configuration; the reference in fig_6_4f.
    """
    means = _config_means(df, dataset)
    best_metric = means["mean_metric"].max()

    return (best_metric - means["mean_metric"]).median()


def regret(df, source, target, retune_lr):
    """Return the regret of using `source`'s best config on `target`.

    Regret is target's best mean metric minus the mean metric of
    source's best configuration on target (zero on the diagonal).

    retune_lr=False transfers the full configuration. retune_lr=True
    transfers only (lora_target, rank, alpha) and picks the best
    learning rate on target.
    """
    source_best = best_config(df, source)
    target_means = _config_means(df, target)
    target_best_metric = target_means["mean_metric"].max()

    if retune_lr:

        candidates = _lookup(target_means, source_best, ["lora_target", "rank", "alpha"])

        assert not candidates.empty, (
            f"transferred config (lora_target={source_best['lora_target']}, "
            f"rank={source_best['rank']}, alpha={source_best['alpha']}) not "
            f"found at all on {target!r} -- the main-sweep grid is supposed "
            f"to be identical across datasets; investigate before trusting "
            f"this result rather than estimating a value"
        )

        transferred_metric = candidates["mean_metric"].max()

    else:

        match = _lookup(target_means, source_best, CONFIG_COLS)

        assert len(match) == 1, (
            f"transferred config {tuple(source_best[CONFIG_COLS])} found "
            f"{len(match)} times on {target!r} (expected exactly 1) -- the "
            f"main-sweep grid is supposed to be identical across datasets; "
            f"investigate before trusting this result rather than "
            f"estimating a value"
        )

        transferred_metric = match["mean_metric"].iloc[0]

    return target_best_metric - transferred_metric


def leave_one_out_default(df, target, datasets):
    """Return the leave-one-out default configuration for `target`.

    The default minimizes the mean regret over all datasets except
    `target`. Returns (config dict, regret on target). The minimum must
    be unique, since fig_6_4f shows it as a single point.
    """
    others = [d for d in datasets if d != target]

    assert others, f"no datasets left after excluding target={target!r} from {datasets!r}"

    regret_frames = []
    for other in others:

        means = _config_means(df, other).set_index(CONFIG_COLS)
        best_metric = means["mean_metric"].max()
        regret_frames.append(best_metric - means["mean_metric"])

    mean_regret = sum(regret_frames) / len(regret_frames)

    min_regret = mean_regret.min()
    n_at_min = (mean_regret == min_regret).sum()

    assert n_at_min == 1, (
        f"leave-one-out argmin for target={target!r} is not unique: "
        f"{n_at_min} configurations tie at mean regret {min_regret} over "
        f"{others!r} -- investigate before presenting a single default"
    )

    loo_key = mean_regret.idxmin()

    target_means = _config_means(df, target).set_index(CONFIG_COLS)
    target_best_metric = target_means["mean_metric"].max()
    loo_metric_on_target = target_means.loc[loo_key, "mean_metric"]

    config = dict(zip(CONFIG_COLS, loo_key))

    return config, target_best_metric - loo_metric_on_target


def percentile_rank(df, source, target):
    """Return the percentile rank (0-100) of source's best on target.

    The configuration keeps its own learning rate. Unlike regret(),
    this does not depend on the metric scale (fig_6_4d).
    """
    source_best = best_config(df, source)
    target_means = _config_means(df, target)

    match = _lookup(target_means, source_best, CONFIG_COLS)

    assert len(match) == 1, (
        f"transferred config {tuple(source_best[CONFIG_COLS])} found "
        f"{len(match)} times on {target!r} (expected exactly 1) -- the "
        f"main-sweep grid is supposed to be identical across datasets; "
        f"investigate before trusting this result rather than estimating "
        f"a value"
    )

    transferred_metric = match["mean_metric"].iloc[0]

    return (target_means["mean_metric"] <= transferred_metric).mean() * 100
