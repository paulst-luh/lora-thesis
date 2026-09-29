"""Expand experiment configs into their runs and print the counts.

Uses the same load_yaml() + generate_experiments() path as
src/experiments/runner.py, but never trains.

Usage:
    python -m scripts.dry_run_configs <config_or_glob> [...]
    python -m scripts.dry_run_configs "configs/mnli/*.yaml"
    python -m scripts.dry_run_configs configs/mnli configs/squad
"""

import argparse
import glob
import os
import sys

from src.utils.config import load_yaml, generate_experiments


def _config_paths(arg):

    if os.path.isdir(arg):
        return sorted(
            p for p in glob.glob(os.path.join(arg, "*.yaml"))
            if not os.path.basename(p).startswith("_")
        )

    matches = sorted(glob.glob(arg))

    if matches:
        return [
            p for p in matches
            if not os.path.basename(p).startswith("_")
        ]

    return [arg]


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "paths",
        nargs="+",
        help="Config file(s), glob pattern(s), or directory/directories "
             "to expand. Files whose name starts with '_' (e.g. _base.yaml, "
             "_task.yaml) are skipped even if matched, since they hold no "
             "grid of their own."
    )

    args = parser.parse_args()

    all_paths = []

    for arg in args.paths:
        all_paths.extend(_config_paths(arg))

    total = 0

    print(f"{'config':55s} {'runs':>6s}")
    print("-" * 62)

    for path in all_paths:

        try:
            config = load_yaml(path)
            experiments = generate_experiments(config)

        except Exception as e:

            print(f"{path:55s} {'ERROR':>6s}  ({e})")
            continue

        n = len(experiments)
        total += n

        print(f"{path:55s} {n:6d}")

    print("-" * 62)
    print(f"{'TOTAL':55s} {total:6d}")


if __name__ == "__main__":
    main()
