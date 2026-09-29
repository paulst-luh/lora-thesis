"""Export the LaTeX tables.

- Appendix A4: runtime and peak memory per dataset and strategy, as in
  fig_6_2b. Zero-shot is excluded because it never trains.
- Section 6.1: parse-failure table, which replaces fig_6_1d because
  only 3 of 1,128 core runs have any parse failure.

Neither is a matplotlib figure, so both bypass the FIGURES registry.

Usage:
    python -m src.analysis.tables
"""

from pathlib import Path

from src.analysis.data import attach_strategy, load_runs
from src.analysis.plots import CORE_BLOCKS, MAIN_DATASET_LABELS, MAIN_DATASETS
from src.analysis.plotstyle import pretty


TABLE_PATH = Path("figures/table_A4_runtime_memory.tex")
PARSE_FAILURE_TABLE_PATH = Path("figures/table_6_1d_parse_failure.tex")

STRATEGIES = ["fft", "qv", "attention", "full"]


def build_runtime_memory_table(df):
    """Return runtime and peak-memory rows per dataset and strategy."""
    core = df[df["block"].isin(CORE_BLOCKS)]
    core = core[core["adaptation"] != "none"]
    core = attach_strategy(core)

    rows = []

    for dataset in MAIN_DATASETS:
        for strategy in STRATEGIES:

            cell = core[(core["dataset"] == dataset) & (core["strategy"] == strategy)]

            rows.append({
                "dataset": MAIN_DATASET_LABELS[dataset],
                "strategy": strategy,
                "runtime_s_mean": cell["runtime_s"].mean(),
                "runtime_s_std": cell["runtime_s"].std(),
                "peak_mem_gb_mean": cell["peak_mem_gb"].mean(),
                "peak_mem_gb_std": cell["peak_mem_gb"].std(),
            })

    return rows


def render_latex_table(rows):
    """Render the runtime/memory rows as a booktabs LaTeX table.

    Written by hand instead of pandas.to_latex() to control the
    "mean +- std" cell format.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Runtime and peak GPU memory per dataset and strategy "
        r"(mean $\pm$ SD over the core grid; wall-clock time is "
        r"hardware-dependent and comparable only within this study).}",
        r"\label{tab:A4_runtime_memory}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Strategy & Runtime (s) & Peak memory (GB) \\",
        r"\midrule",
    ]

    current_dataset = None

    for row in rows:

        dataset_cell = row["dataset"] if row["dataset"] != current_dataset else ""
        current_dataset = row["dataset"]

        runtime = f"{row['runtime_s_mean']:.1f} $\\pm$ {row['runtime_s_std']:.1f}"
        peak_mem = f"{row['peak_mem_gb_mean']:.2f} $\\pm$ {row['peak_mem_gb_std']:.2f}"

        lines.append(f"{dataset_cell} & {pretty(row['strategy'])} & {runtime} & {peak_mem} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def build_parse_failure_table(df):
    """Return per-dataset parse-failure counts for the core grid.

    Each row also holds the worst run's failure rate and metric. MNLI
    is N/A because its constrained-choice scoring never parses text.
    """
    core = df[df["block"].isin(CORE_BLOCKS)]

    rows = []

    for dataset in MAIN_DATASETS:

        sub = core[core["dataset"] == dataset]

        if dataset == "mnli":
            rows.append({
                "dataset": MAIN_DATASET_LABELS[dataset],
                "n_failed": None, "n_total": None,
                "worst_rate": None, "worst_metric": None,
            })
            continue

        failed = sub[sub["parse_failure_rate"] > 0]

        worst_rate = worst_metric = None
        if not failed.empty:
            worst_row = failed.loc[failed["parse_failure_rate"].idxmax()]
            worst_rate = worst_row["parse_failure_rate"]
            worst_metric = worst_row["metric"]

        rows.append({
            "dataset": MAIN_DATASET_LABELS[dataset],
            "n_failed": len(failed), "n_total": len(sub),
            "worst_rate": worst_rate, "worst_metric": worst_metric,
        })

    return rows


def render_parse_failure_table(rows):

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Parser failure rate by dataset, core grid only "
        r"(zeroshot, fft, main\_qv, main\_attention, main\_full; 1{,}128 "
        r"runs). Only 3 of 1{,}128 runs have any parse failure at all -- "
        r"SQuAD's single affected run fails on 100\% of its examples, "
        r"both CoNLL-2003 ones on under 2\% -- and all three are already "
        r"fully collapsed runs (metric 0.0000), so the parse failure is a "
        r"symptom of collapse rather than an independent prompt-formulation "
        r"problem. MNLI is N/A: its constrained-choice scoring never "
        r"invokes a free-text parser, so a zero there would be structural, "
        r"not measured.}",
        r"\label{tab:6_1d_parse_failure}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & Runs with failure & Worst run & Metric of that run \\",
        r"\midrule",
    ]

    for row in rows:

        if row["n_failed"] is None:
            lines.append(f"{row['dataset']} & N/A & --- & --- \\\\")
            continue

        n_failed_cell = f"{row['n_failed']} / {row['n_total']}"

        if row["worst_rate"] is None:
            worst_cell = metric_cell = "---"
        else:
            worst_cell = f"{row['worst_rate'] * 100:.2f}\\%"
            metric_cell = f"{row['worst_metric']:.4f}"

        lines.append(f"{row['dataset']} & {n_failed_cell} & {worst_cell} & {metric_cell} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main():

    df = load_runs()

    rows = build_runtime_memory_table(df)
    latex = render_latex_table(rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(latex)
    print(f"wrote {TABLE_PATH}")

    parse_rows = build_parse_failure_table(df)
    parse_latex = render_parse_failure_table(parse_rows)
    PARSE_FAILURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARSE_FAILURE_TABLE_PATH.write_text(parse_latex)
    print(f"wrote {PARSE_FAILURE_TABLE_PATH}")


if __name__ == "__main__":
    main()
