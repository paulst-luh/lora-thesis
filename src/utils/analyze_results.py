import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "results/thesis_plots"
TABLES_DIR = "results/thesis_tables"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def load_results():

    rows = []

    for root, _, files in os.walk(RESULTS_DIR):

        for file in files:

            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)

            with open(path) as f:
                data = json.load(f)

            config = data["config"]
            metrics = data["metrics"]

            rows.append({

                "experiment":
                    config.get(
                        "experiment_name",
                        "unknown"
                    ),

                "target":
                    config["lora.target"],

                "rank":
                    config["lora.r"],

                "seed":
                    config["seed"],

                "f1":
                    float(metrics["eval_f1"]),

                "accuracy":
                    float(metrics["eval_accuracy"]),

                "loss":
                    float(metrics["eval_loss"]),

                "trainable_params":
                    data["trainable"][
                        "trainable_params"
                    ],

                "trainable_pct":
                    data["trainable"][
                        "trainable_pct"
                    ]
            })

    df = pd.DataFrame(rows)

    return df


def aggregate_results(df):

    agg = (
        df.groupby(
            ["target", "rank"]
        )
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),

            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),

            trainable_params=(
                "trainable_params",
                "mean"
            ),

            trainable_pct=(
                "trainable_pct",
                "mean"
            )
        )
        .reset_index()
    )

    return agg


def save_tables(agg):

    csv_path = os.path.join(
        TABLES_DIR,
        "summary_results.csv"
    )

    agg.to_csv(
        csv_path,
        index=False
    )

    latex_path = os.path.join(
        TABLES_DIR,
        "results_table.tex"
    )

    with open(latex_path, "w") as f:

        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write(
            "Target & Rank & Mean F1 & Std F1 & Trainable Params\\\\\n"
        )
        f.write("\\hline\n")

        for _, row in agg.iterrows():

            f.write(
                f"{row['target']} & "
                f"{int(row['rank'])} & "
                f"{row['mean_f1']:.4f} & "
                f"{row['std_f1']:.4f} & "
                f"{int(row['trainable_params']):,} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def plot_rank_vs_f1(agg):

    plt.figure(figsize=(7,5))

    for target in agg["target"].unique():

        sub = (
            agg[
                agg["target"] == target
            ]
            .sort_values("rank")
        )

        plt.errorbar(
            sub["rank"],
            sub["mean_f1"],
            yerr=sub["std_f1"],
            marker="o",
            capsize=4,
            label=target
        )

    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Mean F1")
    plt.title(
        "LoRA Rank vs Classification Performance"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "rank_vs_f1.png"
        ),
        dpi=300
    )

    plt.close()


def plot_parameter_efficiency(agg):

    plt.figure(figsize=(7,5))

    for target in agg["target"].unique():

        sub = agg[
            agg["target"] == target
        ]

        plt.plot(
            sub["trainable_params"],
            sub["mean_f1"],
            marker="o",
            label=target
        )

    plt.xlabel("Trainable Parameters")
    plt.ylabel("Mean F1")

    plt.title(
        "Parameter Efficiency of LoRA Targets"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "parameter_efficiency.png"
        ),
        dpi=300
    )

    plt.close()


def plot_seed_stability(df):

    stability = (
        df.groupby(
            ["target", "rank"]
        )["f1"]
        .std()
        .reset_index(name="std_f1")
    )

    plt.figure(figsize=(7,5))

    for target in stability["target"].unique():

        sub = (
            stability[
                stability["target"] == target
            ]
            .sort_values("rank")
        )

        plt.plot(
            sub["rank"],
            sub["std_f1"],
            marker="o",
            label=target
        )

    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("F1 Standard Deviation")

    plt.title(
        "Seed Stability Across LoRA Configurations"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "seed_stability.png"
        ),
        dpi=300
    )

    plt.close()


def main():

    df = load_results()

    if df.empty:

        print("No results found.")
        return

    agg = aggregate_results(df)

    print("\n=== AGGREGATED RESULTS ===\n")

    print(agg)

    save_tables(agg)

    plot_rank_vs_f1(agg)

    plot_parameter_efficiency(agg)

    plot_seed_stability(df)

    print(
        f"\nSaved plots to: {PLOTS_DIR}"
    )

    print(
        f"Saved tables to: {TABLES_DIR}"
    )


if __name__ == "__main__":
    main()