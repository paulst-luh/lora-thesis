"""Export the exact HyperSHAP values of the analysis to one CSV.

Reproduces the four sections of notebooks/hypershap_analysis.ipynb with
the same settings. No surrogate model is fitted: the value function is
an exact lookup of the measured, seed-averaged cell values, and the
sensitivity variance is taken over every grid configuration exactly
once. This needs the fully crossed grid (every rank at scaling ratios
1 and 2, i.e. including the (4, 4) and (64, 64) runs).

- Section A: main-sweep sensitivity (FSII, order 2), per dataset.
- Section A ablation: safe (qv, r=4, alpha=8, lr 1e-4) -> risky (full,
  r=64, alpha=128, lr 5e-4), per dataset.
- Section B: dataset-size ladder sensitivity (Attention LoRA), per
  dataset and regime (MNLI and SQuAD only).
- Section C: collapse-probability sensitivity, per dataset, with the
  risky-seed reruns included. GSM8K is kept as an all-zero result.

Baselines of the sensitivity games: the smallest value of every
hyperparameter, with the targets ordered qv < attention < full, i.e.
(qv, r=4, ratio 1, lr 1e-4) for the main sweep and collapse and
(r=4, 1,000 examples, lr 1e-4) for the ladder.

Every configuration space is checked to match its data one to one, and
every game is checked before anything is written (NUMERICAL_ZERO).

Usage:
    python -m scripts.export_hypershap_values
"""

import glob
import itertools
import json

import hypershap.hypershap as hs_core
import numpy as np
import pandas as pd
from ConfigSpace import CategoricalHyperparameter, Configuration, ConfigurationSpace, OrdinalHyperparameter
from hypershap import ExplanationTask, HyperSHAP
from hypershap.games import AblationGame
from hypershap.task import AblationExplanationTask
from hypershap.utils import RandomConfigSpaceSearcher


FINAL_CSV = "results/prompting_study_final.csv"
OUTPUT_CSV = "results/hypershap_values.csv"

MAIN_SWEEP_BLOCKS = ["main_qv", "main_attention", "main_full"]
MAIN_DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
LADDER_DATASETS = ["mnli", "squad"]
LADDER_REGIMES = ["ladder_isoepoch", "ladder_isostep"]

# Exact grid values; every configuration space is built from these.
MAIN_GRID = {
    "rank": [4, 8, 16, 32, 64],
    "scaling_ratio": [1.0, 2.0],
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "lora_target": ["qv", "attention", "full"],
}
LADDER_GRID = {
    "subset_size": [1000, 5000, 20000, 50000],
    "rank": [4, 8, 16, 32, 64],
    "learning_rate": [1e-4, 3e-4],
}
CATEGORICAL_COLS = ["lora_target"]

# Sensitivity baselines: the smallest value of every hyperparameter
# (targets ordered qv < attention < full), which is also each space's
# default configuration. Passed explicitly so they stay fixed.
MAIN_BASELINE = {"rank": 4, "scaling_ratio": 1.0, "learning_rate": 1e-4, "lora_target": "qv"}
LADDER_BASELINE = {"subset_size": 1000, "rank": 4, "learning_rate": 1e-4}

SAFE_CONFIG = {"rank": 4, "scaling_ratio": 2.0, "learning_rate": 1e-4, "lora_target": "qv"}
RISKY_CONFIG = {"rank": 64, "scaling_ratio": 2.0, "learning_rate": 5e-4, "lora_target": "full"}

# value(risky) - value(safe) per dataset, from the measured data.
EXPECTED_ABLATION_GAP = {
    "mnli": -0.711752, "squad": -0.843433, "conll2003": -0.931007, "gsm8k": -0.116000,
}

# Numerical zero for the checks. shapiq solves FSII by least squares
# (np.linalg.lstsq), which leaves residuals of up to about 1.5e-9 even
# for an exact game; the thesis reports values to 1e-6.
NUMERICAL_ZERO = 1e-8

# Largest residual seen per check, printed at the end.
MAX_RESIDUAL = {"a": 0.0, "b": 0.0, "c2": 0.0}

# Risky-seed reruns, merged into the main sweep for the collapse rates
# (n=10 cells), as in the notebook.
RISKYSEEDS_DIRS = [
    "results/conll2003/prompting_main_full_riskyseeds",
    "results/mnli/prompting_main_full_riskyseeds_lr5e4",
    "results/mnli/prompting_main_full_riskyseeds_lr3e4",
    "results/squad/prompting_main_full_riskyseeds_lr5e4",
    "results/squad/prompting_main_full_riskyseeds_lr3e4",
    "results/conll2003/prompting_main_attention_riskyseeds",
    "results/mnli/prompting_main_attention_riskyseeds_lr5e4",
    "results/mnli/prompting_main_attention_riskyseeds_lr3e4",
    "results/squad/prompting_main_attention_riskyseeds_lr3e4",
]


class ExactLookup:
    """Returns the measured value of a configuration; never estimates one."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        keys = [tuple(np.round(row, 12)) for row in X]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate configurations: aggregate over seeds first")
        self.table_ = dict(zip(keys, y))
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        try:
            return np.array([self.table_[tuple(np.round(r, 12))] for r in X])
        except KeyError as e:
            raise KeyError(f"configuration was not measured: {e}") from None

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y, float)))


class GridSearcher(RandomConfigSpaceSearcher):
    """Evaluates every grid configuration exactly once, no sampling."""

    grid = None  # Set to the data rows' configuration vectors before each call.

    def __init__(self, explanation_task, **kwargs):
        # The drawn samples are replaced below. ConfigSpace 1.2 returns
        # a single Configuration instead of a list for size=1, hence 2.
        kwargs["n_samples"] = 2
        super().__init__(explanation_task, **kwargs)
        self.random_sample = np.array(GridSearcher.grid, copy=True)
        self.coalition_cache = {}


hs_core.RandomConfigSpaceSearcher = GridSearcher


def _to_native(col, value):
    # ConfigSpace needs one consistent type per hyperparameter.
    if col in ("rank", "subset_size"):
        return int(value)
    if col in CATEGORICAL_COLS:
        return value
    return float(value)


def _build_space(grid):
    cs = ConfigurationSpace()
    for col, values in grid.items():
        if col in CATEGORICAL_COLS:
            cs.add(CategoricalHyperparameter(col, choices=values))
        else:
            cs.add(OrdinalHyperparameter(col, sequence=values))
    return cs


def _build_task(agg, grid, value_col):
    """Build an exact-lookup task, one data row per grid configuration.

    Returns (ExplanationTask, data), data being the (Configuration,
    value) pairs the lookup table is built from.
    """
    cs = _build_space(grid)
    cols = list(grid)
    data = [
        (
            Configuration(cs, values={col: _to_native(col, getattr(row, col)) for col in cols}),
            float(getattr(row, value_col)),
        )
        for row in agg.itertuples()
    ]

    n_space = int(np.prod([len(values) for values in grid.values()]))
    assert len(data) == n_space, f"space has {n_space} configurations, data {len(data)} rows"
    present = {tuple(_to_native(col, getattr(row, col)) for col in cols) for row in agg.itertuples()}
    missing = [combo for combo in itertools.product(*grid.values()) if combo not in present]
    assert not missing, f"configurations without data: {missing}"

    task = ExplanationTask.from_data(config_space=cs, data=data, base_model=ExactLookup())
    return task, data


def _order_sum(hs, iv):
    """Return the sum of all order-1 and order-2 values."""
    named = hs.get_interaction_values_with_names(iv)
    return sum(value for names, value in named.items() if len(names) in (1, 2))


def _sensitivity(task, data, baseline):
    """Return (HyperSHAP, values) of the exact sensitivity game.

    Checks efficiency (check b): the order-1 and order-2 values sum to
    the variance of the measured values over the whole grid.
    """
    cs = task.config_space
    baseline_config = Configuration(cs, values=baseline)
    assert baseline_config == cs.get_default_configuration(), "baseline differs from the space default"

    GridSearcher.grid = np.array([cfg.get_array() for cfg, _ in data])
    hs = HyperSHAP(task)
    iv = hs.sensitivity(baseline_config=baseline_config, index="FSII", order=2)

    # Check b: the order-1 and order-2 values sum to the variance.
    variance = np.var([value for _, value in data])
    residual = abs(_order_sum(hs, iv) - variance)
    MAX_RESIDUAL["b"] = max(MAX_RESIDUAL["b"], residual)
    assert residual <= NUMERICAL_ZERO, f"check b: residual {residual!r}"
    return hs, iv


def _measured(data, config):
    return next(value for cfg, value in data if cfg == config)


def _check_null_player(task, safe, risky, player, dataset):
    """Check c1: v(S + {player}) == v(S) exactly in the ablation game.

    Uses the same AblationGame as HyperSHAP.ablation() and its raw
    lookup values, for every coalition S without `player`.
    """
    game = AblationGame(explanation_task=AblationExplanationTask(
        config_space=task.config_space,
        surrogate_model=task.get_single_surrogate_model(),
        baseline_config=safe,
        config_of_interest=risky,
    ))
    names = list(task.config_space.keys())
    j = names.index(player)
    for bits in itertools.product([False, True], repeat=len(names)):
        if bits[j]:
            continue
        without = np.array(bits)
        with_player = without.copy()
        with_player[j] = True
        v_without = game.evaluate_single_coalition(without)
        v_with = game.evaluate_single_coalition(with_player)
        assert v_with == v_without, f"check c1, {dataset}: {player} not null for {without}"


def _rows_from_iv(hs, iv, section, dataset, regime=None):
    """Return the values of `iv` as tidy rows for the output CSV.

    Includes main effects, pairwise interactions and the baseline.
    """
    rows = []

    for names, value in hs.get_interaction_values_with_names(iv).items():
        rows.append({
            "section": section,
            "dataset": dataset,
            "regime": regime,
            "order": len(names),
            "players": "+".join(names) if names else "(baseline)",
            "value": value,
        })

    return rows


def main():

    df = pd.read_csv(FINAL_CSV)
    df["block"] = df["block"].str.replace(r"_ratio1_extend$", "", regex=True)
    main_sweep_df = df[df["block"].isin(MAIN_SWEEP_BLOCKS)]
    main_cols = list(MAIN_GRID)

    all_rows = []

    # Section A: main-sweep sensitivity per dataset, one row per
    # configuration (mean over its seeds).
    main_tasks = {}

    for dataset in MAIN_DATASETS:

        subset = main_sweep_df[main_sweep_df["dataset"] == dataset]
        agg = subset.groupby(main_cols, as_index=False)["metric"].mean()
        task, data = _build_task(agg, MAIN_GRID, "metric")
        hs, iv = _sensitivity(task, data, MAIN_BASELINE)

        main_tasks[dataset] = (hs, data)
        all_rows += _rows_from_iv(hs, iv, "main_sweep_sensitivity", dataset)

        print(f"main_sweep_sensitivity/{dataset}: {len(data)} configurations, efficiency OK")

    # Section A ablation: safe -> risky, on each dataset's space.
    for dataset, (hs, data) in main_tasks.items():

        cs = hs.explanation_task.config_space
        safe = Configuration(cs, values=SAFE_CONFIG)
        risky = Configuration(cs, values=RISKY_CONFIG)

        # Check c1: in the game itself (lookup values, before any
        # solve), scaling_ratio is a null player: both endpoints use
        # ratio 2.
        _check_null_player(hs.explanation_task, safe, risky, "scaling_ratio", dataset)

        iv = hs.ablation(config_of_interest=risky, baseline_config=safe, index="FSII", order=2)

        # Check a: the order-1 and order-2 values sum to the gap.
        gap = _measured(data, risky) - _measured(data, safe)
        assert abs(gap - EXPECTED_ABLATION_GAP[dataset]) < 5e-7, f"{dataset}: measured gap {gap!r}"
        residual = abs(_order_sum(hs, iv) - gap)
        MAX_RESIDUAL["a"] = max(MAX_RESIDUAL["a"], residual)
        assert residual <= NUMERICAL_ZERO, f"check a, {dataset}: residual {residual!r}"

        # Check c2: every FSII term containing scaling_ratio is zero.
        named = hs.get_interaction_values_with_names(iv)
        for names, value in named.items():
            if "scaling_ratio" in names:
                MAX_RESIDUAL["c2"] = max(MAX_RESIDUAL["c2"], abs(value))
                assert abs(value) <= NUMERICAL_ZERO, f"check c2, {dataset}: {names} = {value!r}"

        all_rows += _rows_from_iv(hs, iv, "ablation_safe_to_risky", dataset)

        print(f"ablation_safe_to_risky/{dataset}: measured gap {gap:+.6f}, checks a, c1, c2 OK")

    # Section B: ladder sensitivity (Attention LoRA, MNLI and SQuAD).
    ladder_cols = list(LADDER_GRID)

    for dataset in LADDER_DATASETS:
        for regime in LADDER_REGIMES:

            subset = df[(df["dataset"] == dataset) & (df["block"] == regime)]
            agg = subset.groupby(ladder_cols, as_index=False)["metric"].mean()
            task, data = _build_task(agg, LADDER_GRID, "metric")
            hs, iv = _sensitivity(task, data, LADDER_BASELINE)

            all_rows += _rows_from_iv(hs, iv, "ladder_sensitivity", dataset, regime=regime)

            print(f"ladder_sensitivity/{dataset}/{regime}: {len(data)} configurations, efficiency OK")

    # Section C: collapse-probability sensitivity, risky-seed reruns
    # included. Collapse means metric < 0.3 as in the notebook; this is
    # separate from is_collapsed() in src/analysis/data.py.
    extra_rows = []
    for riskydir in RISKYSEEDS_DIRS:
        for f in glob.glob(f"{riskydir}/metrics/*.json"):
            row = json.load(open(f))["row"]
            extra_rows.append({
                "dataset": row["dataset"], "lora_target": row["lora_target"], "rank": row["rank"],
                "scaling_ratio": row["scaling_ratio"], "learning_rate": row["learning_rate"], "metric": row["metric"],
            })
    extra_df = pd.DataFrame(extra_rows)

    collapse_input = pd.concat([
        main_sweep_df[["dataset", "lora_target", "rank", "scaling_ratio", "learning_rate", "metric"]],
        extra_df,
    ], ignore_index=True)

    # CSV-sourced learning_rate is float, JSON-sourced is string.
    for col in ["rank", "scaling_ratio", "learning_rate"]:
        collapse_input[col] = pd.to_numeric(collapse_input[col])

    collapse_cell_cols = ["dataset", "lora_target", "rank", "scaling_ratio", "learning_rate"]
    collapse_rates = collapse_input.groupby(collapse_cell_cols).agg(
        n=("metric", "size"),
        collapse_rate=("metric", lambda s: (s < 0.3).mean()),
    ).reset_index()

    # The risky-seed reruns only add seeds to existing cells; a merge or
    # dtype bug would show up as extra cells.
    assert len(collapse_rates) == 90 * 4, f"expected 360 cells, got {len(collapse_rates)}"

    for dataset in MAIN_DATASETS:

        sub = collapse_rates[collapse_rates["dataset"] == dataset]
        task, data = _build_task(sub, MAIN_GRID, "collapse_rate")
        hs, iv = _sensitivity(task, data, MAIN_BASELINE)

        all_rows += _rows_from_iv(hs, iv, "collapse_sensitivity", dataset)

        print(f"collapse_sensitivity/{dataset}: {len(data)} cells ({(sub['n'] == 10).sum()} with n=10), efficiency OK")

    # Check d: ExactLookup raises on any unmeasured configuration, and
    # hypershap does not catch it, so reaching this point means none was
    # requested.
    print("\nmaximum residuals: " + ", ".join(f"{k} {v:.2e}" for k, v in MAX_RESIDUAL.items())
          + f" (limit {NUMERICAL_ZERO:g}); c1 exact; d no unmeasured configuration")

    out = pd.DataFrame(all_rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nwrote {OUTPUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
