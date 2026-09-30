"""Rebuild results/prompting_study_final.csv.

Keeps the four prompting experiment groups of
results/runs_prompting_study.csv and derives the `block` column from
experiment_name (docs/experimental_design_context.md §3.8), merging the
blocks in MERGED_BLOCKS into their parent block. Run
`python -m src.utils.aggregate_runs` first.

Usage:
    python -m scripts.build_final_csv
"""

import pandas as pd


RUNS_CSV = "results/runs_prompting_study.csv"
FINAL_CSV = "results/prompting_study_final.csv"

PROMPTING_GROUPS = [
    "mnli_prompting", "squad_prompting", "conll2003_prompting", "gsm8k_prompting",
]

# Runs that complete a main-sweep grid (the ratio-1 pairs (4, 4) and
# (64, 64)) join that block; experiment_name keeps their origin.
MERGED_BLOCKS = {
    "main_qv_ratio1_extend": "main_qv",
    "main_attention_ratio1_extend": "main_attention",
    "main_full_ratio1_extend": "main_full",
}


def main():

    df = pd.read_csv(RUNS_CSV)
    df = df[df["experiment_group"].isin(PROMPTING_GROUPS)].copy()
    df["block"] = df["experiment_name"].str.replace(
        r"^(mnli|squad|conll2003|gsm8k)_prompting_", "", regex=True
    ).replace(MERGED_BLOCKS)

    df.to_csv(FINAL_CSV, index=False)

    print(f"wrote {FINAL_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
