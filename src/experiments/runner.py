from src.utils.config import load_yaml, generate_experiments

from src.train import train


def run_experiment(
    config_path,
    index=None
):

    print("STARTING EXPERIMENT")

    config = load_yaml(config_path)

    experiments = generate_experiments(config)

    print(
        f"Total experiments: "
        f"{len(experiments)}"
    )

    # Array mode: run only experiment `index` (one SLURM array task).
    if index is not None:

        index = int(index)

        if index >= len(experiments):

            raise ValueError(
                f"Index {index} "
                f"out of range "
                f"({len(experiments)})"
            )

        print(
            f"\n--- Running experiment "
            f"{index} ---"
        )

        train(experiments[index])

    else:

        # Serial mode: run every experiment in order.
        for i, exp in enumerate(experiments):

            print(
                f"\n--- Experiment "
                f"{i+1}/{len(experiments)} ---"
            )

            train(exp)


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 2:

        run_experiment(
            sys.argv[1]
        )

    elif len(sys.argv) == 3:

        run_experiment(
            sys.argv[1],
            sys.argv[2]
        )

    else:

        print(
            "Usage:\n"
            "python -m src.experiments.runner <config>\n"
            "python -m src.experiments.runner <config> <index>"
        )