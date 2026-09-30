"""Write the HyperSHAP LaTeX tables from results/hypershap_values.csv.

- Table 6.4 (tab:hypershap-main-sweep): main-sweep sensitivity.
- Table A.4 (tab:hypershap-ladder): dataset-size ladder sensitivity.
- Table A.5 (tab:hypershap-collapse): collapse-probability sensitivity.

Values are shown in units of 10^-3 with the display rule of
src/analysis/plotstyle.py (fmt_e3), in the layout of the thesis source.
Nothing is computed here; run scripts/export_hypershap_values.py first.

Usage:
    python -m scripts.hypershap_tables
"""

from pathlib import Path

import pandas as pd

from src.analysis.plotstyle import fmt_e3


VALUES_CSV = "results/hypershap_values.csv"

MAIN_HEADER = [r"& MNLI & SQuAD & CoNLL-2003 & GSM8K \\"]
LADDER_HEADER = [
    r"& \multicolumn{2}{c}{MNLI} & \multicolumn{2}{c}{SQuAD} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
    r"& iso-epoch & iso-step & iso-epoch & iso-step \\",
]

# (dataset, regime) per column, in table order.
MAIN_COLUMNS = [("mnli", None), ("squad", None), ("conll2003", None), ("gsm8k", None)]
LADDER_COLUMNS = [("mnli", "ladder_isoepoch"), ("mnli", "ladder_isostep"),
                  ("squad", "ladder_isoepoch"), ("squad", "ladder_isostep")]

# (players, row label) per order, in table order.
MAIN_ROWS = {
    1: [("learning_rate", r"\ac{LR}"),
        ("lora_target", r"\ac{TM}"),
        ("rank", r"Rank $r$"),
        ("scaling_ratio", r"Scaling ratio $\alpha/r$")],
    2: [("learning_rate+lora_target", r"$LR$ $\times$ $TM$"),
        ("learning_rate+rank", r"$LR$ $\times$ $r$"),
        ("learning_rate+scaling_ratio", r"$LR$ $\times$ $\alpha/r$"),
        ("lora_target+rank", r"$TM$ $\times$ $r$"),
        ("lora_target+scaling_ratio", r"$TM$ $\times$ $\alpha/r$"),
        ("rank+scaling_ratio", r"$r$ $\times$ $\alpha/r$")],
}
LADDER_ROWS = {
    1: [("learning_rate", "Learning rate"),
        ("rank", r"Rank $r$"),
        ("subset_size", "Training set size")],
    2: [("learning_rate+rank", r"Learning rate $\times$ Rank $r$"),
        ("learning_rate+subset_size", r"Learning rate $\times$ Training set size"),
        ("rank+subset_size", r"Rank $r$ $\times$ Training set size")],
}

UNITS = r"Values are given in units of $10^{-3}$."

# One entry per table: section, header, columns, rows, caption, label,
# output file (named after the label, not the table number).
TABLES = [
    {
        "section": "main_sweep_sensitivity",
        "header": MAIN_HEADER, "columns": MAIN_COLUMNS, "rows": MAIN_ROWS,
        "caption": "Faithful Shapley Interaction Indices of the main sweep, "
                   "computed separately for each dataset. " + UNITS,
        "label": "tab:hypershap-main-sweep",
        "path": "figures/table_hypershap_main_sweep.tex",
    },
    {
        "section": "ladder_sensitivity",
        "header": LADDER_HEADER, "columns": LADDER_COLUMNS, "rows": LADDER_ROWS,
        "caption": r"Faithful Shapley Interaction Indices of the two ladder blocks of "
                   r"Section \ref{sec:6.3}, with the \ac{TM} fixed at Attention \ac{LoRA}. " + UNITS,
        "label": "tab:hypershap-ladder",
        "path": "figures/table_hypershap_ladder.tex",
    },
    {
        "section": "collapse_sensitivity",
        "header": MAIN_HEADER, "columns": MAIN_COLUMNS, "rows": MAIN_ROWS,
        "caption": "Faithful Shapley Interaction Indices with the collapse rate of a "
                   "cell as the target instead of the mean score. " + UNITS,
        "label": "tab:hypershap-collapse",
        "path": "figures/table_hypershap_collapse.tex",
    },
]


def tex_number(value):
    """Return fmt_e3(value) with LaTeX minus and less-than signs."""
    text = fmt_e3(value)
    if text.startswith("<"):
        return "$<$" + text[1:]
    if text.startswith("-"):
        return "$-$" + text[1:]
    return text


def lookup(values, section, dataset, regime, players):
    """Return the value of a term; fail if it is missing or repeated."""
    match = values[
        (values["section"] == section)
        & (values["dataset"] == dataset)
        & (values["regime"].fillna("") == (regime or ""))
        & (values["players"].map(lambda p: frozenset(p.split("+"))) == frozenset(players.split("+")))
    ]
    assert len(match) == 1, f"{section}/{dataset}/{regime}/{players}: {len(match)} rows"
    return match["value"].iloc[0]


def render(values, table):
    """Render one table in the layout of the thesis source."""
    n_cols = len(table["columns"]) + 1
    width = max(len(label) for rows in table["rows"].values() for _, label in rows)

    body = [r"\toprule", *table["header"]]
    for order in (1, 2):
        body += [r"\midrule", rf"\multicolumn{{{n_cols}}}{{l}}{{\emph{{Order {order}}}}} \\"]
        for players, label in table["rows"][order]:
            cells = [tex_number(lookup(values, table["section"], dataset, regime, players))
                     for dataset, regime in table["columns"]]
            body.append(f"{label:<{width}} & " + " & ".join(f"{cell:>8}" for cell in cells) + r" \\")
    body.append(r"\bottomrule")

    lines = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "r" * len(table["columns"]) + "}",
        *(f"    {line}" for line in body),
        r"  \end{tabular}",
        rf"  \caption{{{table['caption']}}}",
        rf"  \label{{{table['label']}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():

    values = pd.read_csv(VALUES_CSV)

    for table in TABLES:
        path = Path(table["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(values, table))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
