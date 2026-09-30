"""Exact Shapley values and pairwise interactions of hyperparameters.

The grid was run completely, so every game is enumerated exactly; no
surrogate model is fitted (the HyperSHAP analysis likewise uses the
measured values).

Game definition:

- Players: the hyperparameters. The main-sweep game uses lora_target,
  learning_rate and (rank, alpha) as one player, because the pairs
  were run as fixed pairs (10 pairs: ranks 4-64, each at scaling
  ratios 1 and 2). The ladder game uses rank, subset_size,
  learning_rate and training_regime; the rank vs. ratio game uses
  rank and scaling_ratio.
- v(S): the best seed-mean metric when the players in S are optimized
  over their grid values and all other players stay at DEFAULTS.
- Shapley value: weighted mean of v(S + {i}) - v(S) over coalitions S
  without i, with weight |S|!(n-|S|-1)!/n!. The efficiency axiom is
  asserted in shapley_values().
- Interaction: the Grabisch-Roubens index (1999), a weighted mean of
  v(S+{i,j}) - v(S+{i}) - v(S+{j}) + v(S) over coalitions S without
  i and j, with weight |S|!(n-|S|-2)!/(n-1)!.

Defaults: lora_target "qv" (the classic baseline), learning_rate 3e-4
(middle of the main grid and part of the ladder grid), (rank, alpha)
(8, 16) (ratio 2, the common convention; the rank and scaling_ratio
defaults follow from it), subset_size 5000 and training_regime
"iso_epoch".
"""

import itertools
from math import factorial

import pandas as pd


# Constants

MAIN_SWEEP_BLOCKS = ["main_qv", "main_attention", "main_full"]
LADDER_BLOCKS = ["ladder_isoepoch", "ladder_isostep"]

MAIN_DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
LADDER_DATASETS = ["mnli", "squad"]

RANK_ALPHA_PAIRS = [
    (4, 4), (4, 8), (8, 8), (8, 16), (16, 16),
    (16, 32), (32, 32), (32, 64), (64, 64), (64, 128),
]

DEFAULTS = {
    "lora_target": "qv",
    "learning_rate": 3e-4,
    "rank_alpha": (8, 16),
    "rank": 8,
    "scaling_ratio": 2,
    "subset_size": 5000,
    "training_regime": "iso_epoch",
}

MAIN_CONTRIBUTORS = {
    "lora_target": {
        "values": ["qv", "attention", "full"],
        "columns": lambda v: {"lora_target": v},
    },
    "learning_rate": {
        "values": [1e-4, 3e-4, 5e-4],
        "columns": lambda v: {"learning_rate": v},
    },
    "rank_alpha": {
        "values": RANK_ALPHA_PAIRS,
        "columns": lambda v: {"rank": v[0], "alpha": v[1]},
    },
}

LADDER_CONTRIBUTORS = {
    "rank": {
        "values": [4, 8, 16, 32, 64],
        "columns": lambda v: {"rank": v},
    },
    "subset_size": {
        "values": [1000, 5000, 20000, 50000],
        "columns": lambda v: {"subset_size": v},
    },
    "learning_rate": {
        "values": [1e-4, 3e-4],
        "columns": lambda v: {"learning_rate": v},
    },
    "training_regime": {
        "values": ["iso_epoch", "iso_step"],
        "columns": lambda v: {"training_regime": v},
    },
}

RANK_RATIO_CONTRIBUTORS = {
    "rank": {
        "values": [4, 8, 16, 32, 64],
        "columns": lambda v: {"rank": v},
    },
    "scaling_ratio": {
        "values": [1, 2],
        "columns": lambda v: {"scaling_ratio": v},
    },
}


# Characteristic function

def _restricted_rows(df, blocks, dataset, fixed_filters=None):

    sub = df[(df["block"].isin(blocks)) & (df["dataset"] == dataset)]

    if fixed_filters:
        for col, val in fixed_filters.items():
            sub = sub[sub[col] == val]

    return sub


def _score(rows, column_filter):
    """Return the mean metric of the rows matching `column_filter`."""
    mask = pd.Series(True, index=rows.index)

    for col, val in column_filter.items():
        mask &= rows[col] == val

    matched = rows.loc[mask, "metric"]

    assert len(matched) > 0, (
        f"no rows found for {column_filter} -- the grid is assumed complete "
        f"(every combination was actually run); a missing combination means "
        f"the filter or the assumed grid is wrong, not something to estimate"
    )

    return matched.mean()


def characteristic_function(rows, contributors, defaults, coalition):
    """Return v(S): the best mean metric, optimizing `coalition`.

    Contributors outside `coalition` are held at `defaults`.
    """
    free = [name for name in contributors if name in coalition]
    fixed = [name for name in contributors if name not in coalition]

    fixed_filter = {}
    for name in fixed:
        fixed_filter.update(contributors[name]["columns"](defaults[name]))

    if not free:
        return _score(rows, fixed_filter)

    free_value_lists = [contributors[name]["values"] for name in free]

    best = float("-inf")

    for combo in itertools.product(*free_value_lists):

        row_filter = dict(fixed_filter)

        for name, value in zip(free, combo):
            row_filter.update(contributors[name]["columns"](value))

        best = max(best, _score(rows, row_filter))

    return best


def all_coalition_values(rows, contributors, defaults):
    """Return v(S) for each coalition S, keyed by frozenset of names."""
    names = list(contributors.keys())
    values = {}

    for size in range(len(names) + 1):
        for combo in itertools.combinations(names, size):

            coalition = frozenset(combo)
            values[coalition] = characteristic_function(rows, contributors, defaults, coalition)

    return values


# Shapley value and Grabisch-Roubens interaction index

def shapley_values(v, players):
    """Return the exact Shapley value of each player.

    `v` maps every coalition (frozenset) to its value. The efficiency
    axiom (sum equals v(full) - v(empty)) is asserted.
    """
    n = len(players)
    full = frozenset(players)
    empty = frozenset()

    phi = {}

    for player in players:

        others = [p for p in players if p != player]
        total = 0.0

        for size in range(len(others) + 1):
            for combo in itertools.combinations(others, size):

                s = frozenset(combo)
                weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                total += weight * (v[s | {player}] - v[s])

        phi[player] = total

    efficiency_gap = abs(sum(phi.values()) - (v[full] - v[empty]))

    assert efficiency_gap < 1e-9, (
        f"Shapley efficiency axiom violated: sum(phi)={sum(phi.values())!r} "
        f"but v(full)-v(empty)={v[full] - v[empty]!r} (gap={efficiency_gap!r}) "
        f"-- this indicates a bug in the weighting, not a numerical rounding "
        f"issue at this tolerance"
    )

    return phi


def interaction_index(v, players):
    """Return the Grabisch-Roubens index for every pair of players.

    `v` maps every coalition (frozenset) to its value. Needs at least
    two players.
    """
    n = len(players)
    interactions = {}

    for i, j in itertools.combinations(players, 2):

        others = [p for p in players if p not in (i, j)]
        total = 0.0

        for size in range(len(others) + 1):
            for combo in itertools.combinations(others, size):

                s = frozenset(combo)
                weight = factorial(len(s)) * factorial(n - len(s) - 2) / factorial(n - 1)
                delta = v[s | {i, j}] - v[s | {i}] - v[s | {j}] + v[s]
                total += weight * delta

        interactions[(i, j)] = total

    return interactions


# Per-game entry points

def main_sweep_attribution(df, dataset):
    """Return (shapley, interactions) of the main-sweep game.

    Players: lora_target, learning_rate and rank_alpha.
    """
    rows = _restricted_rows(df, MAIN_SWEEP_BLOCKS, dataset)
    v = all_coalition_values(rows, MAIN_CONTRIBUTORS, DEFAULTS)
    players = list(MAIN_CONTRIBUTORS.keys())

    return shapley_values(v, players), interaction_index(v, players)


def ladder_attribution(df, dataset):
    """Return (shapley, interactions) of the ladder game.

    Players: rank, subset_size, learning_rate and training_regime.
    MNLI and SQuAD only; lora_target is always "attention" there.
    """
    rows = _restricted_rows(df, LADDER_BLOCKS, dataset, fixed_filters={"lora_target": "attention"})
    v = all_coalition_values(rows, LADDER_CONTRIBUTORS, DEFAULTS)
    players = list(LADDER_CONTRIBUTORS.keys())

    return shapley_values(v, players), interaction_index(v, players)


def rank_ratio_attribution(df, dataset):
    """Return (shapley, interactions) of the rank vs. ratio game.

    Uses all five ranks, since every rank has both ratios 1 and 2 in
    the main sweep, so rank and ratio vary independently.
    lora_target and learning_rate are held at DEFAULTS.
    """
    rows = _restricted_rows(
        df, MAIN_SWEEP_BLOCKS, dataset,
        fixed_filters={
            "lora_target": DEFAULTS["lora_target"],
            "learning_rate": DEFAULTS["learning_rate"],
        },
    )
    rows = rows[rows["rank"].isin(RANK_RATIO_CONTRIBUTORS["rank"]["values"])]

    v = all_coalition_values(rows, RANK_RATIO_CONTRIBUTORS, DEFAULTS)
    players = list(RANK_RATIO_CONTRIBUTORS.keys())

    return shapley_values(v, players), interaction_index(v, players)


# Tidy-DataFrame wrappers for the figure functions

def _tidy(attribution_fn, df, datasets):

    shapley_rows = []
    interaction_rows = []

    for dataset in datasets:

        shapley, interactions = attribution_fn(df, dataset)

        for name, value in shapley.items():
            shapley_rows.append({"dataset": dataset, "contributor": name, "shapley_value": value})

        for (i, j), value in interactions.items():
            interaction_rows.append({"dataset": dataset, "i": i, "j": j, "interaction": value})

    return pd.DataFrame(shapley_rows), pd.DataFrame(interaction_rows)


def all_main_sweep_attributions(df, datasets=MAIN_DATASETS):
    return _tidy(main_sweep_attribution, df, datasets)


def all_ladder_attributions(df, datasets=LADDER_DATASETS):
    return _tidy(ladder_attribution, df, datasets)


def all_rank_ratio_attributions(df, datasets=MAIN_DATASETS):
    return _tidy(rank_ratio_attribution, df, datasets)
