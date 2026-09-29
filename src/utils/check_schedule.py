"""Check the iso_epoch / iso_step training schedules.

For each rung of a dataset-size ladder, build the real
TrainingArguments via build_training_args() and step a real LR
scheduler through the resolved number of optimizer steps. No model,
data or GPU is needed.

The check fails if iso_step no longer gives the same step count on
every rung (e.g. a transformers update stops letting max_steps
override the epoch count), or if the final LR differs across rungs.

Usage:
    python -m src.utils.check_schedule
    python -m src.utils.check_schedule --config <config.yaml>
"""

import argparse
import math

import torch
from transformers import get_scheduler

from src.utils.config import load_yaml, generate_experiments
from src.utils.training import build_training_args


DEFAULT_LADDER = [1000, 5000, 20000, 50000]

# Mirrors the standard fixed: block, so the check runs without a
# config file. Use --config to check a real one.
DEFAULT_CONFIG = {

    "output.root": "results/debug/check_schedule",

    "training.optimizer": "paged_adamw_8bit",
    "training.weight_decay": 0.01,
    "training.learning_rate": 2e-4,
    "training.warmup_ratio": 0.03,
    "training.lr_scheduler_type": "cosine",
    "training.batch_size": 8,
    "training.grad_accum": 2,
    "training.epochs": 3,

    "pipeline.type": "head",

}


def steps_per_epoch(subset_size, batch_size, grad_accum):
    """Return optimizer steps per epoch, as Trainer computes them.

    Batches per epoch divided by grad_accum, rounded up (single
    process).
    """
    len_dataloader = math.ceil(subset_size / batch_size)

    return max(
        len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0),
        1
    )


def resolved_max_steps(args, subset_size):
    """Return (max_steps, steps_per_epoch) as Trainer resolves them.

    A positive max_steps wins; otherwise it follows from the epochs.
    """
    per_epoch = steps_per_epoch(
        subset_size,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps
    )

    if args.max_steps > 0:
        return args.max_steps, per_epoch

    return math.ceil(args.num_train_epochs * per_epoch), per_epoch


def run_ladder(config, ladder):

    regime = config.get("training.regime", "iso_epoch")

    print(f"\n=== regime = {regime} ===")

    header = f"{'subset_size':>12} | {'steps/epoch':>11} | {'optimizer_steps':>15} | {'final_lr':>12}"

    print(header)
    print("-" * len(header))

    seen_steps = set()
    seen_final_lr = set()

    for subset_size in ladder:

        run_config = dict(config)
        run_config["dataset.subset_size"] = subset_size

        args = build_training_args(run_config, seed=42)

        max_steps, per_epoch = resolved_max_steps(args, subset_size)

        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([dummy_param], lr=args.learning_rate)

        scheduler = get_scheduler(

            args.lr_scheduler_type,

            optimizer=optimizer,

            num_warmup_steps=args.get_warmup_steps(max_steps),

            num_training_steps=max_steps

        )

        for _ in range(max_steps):
            optimizer.step()
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]

        seen_steps.add(max_steps)
        seen_final_lr.add(round(final_lr, 10))

        print(f"{subset_size:>12} | {per_epoch:>11} | {max_steps:>15} | {final_lr:>12.6e}")

    if regime == "iso_step":

        if len(seen_steps) != 1:
            print(f"FAIL: optimizer_steps is not constant across the ladder: {sorted(seen_steps)}")
        else:
            print(f"OK: optimizer_steps constant at {seen_steps.pop()} across every rung")

    elif regime == "iso_epoch":

        if len(seen_steps) == 1:
            print("FAIL: optimizer_steps did NOT scale with subset_size -- expected growth across the ladder")
        else:
            print(f"OK: optimizer_steps scales with subset_size: {sorted(seen_steps)}")

    if len(seen_final_lr) != 1:
        print(f"FAIL: final LR differs across the ladder within one regime: {sorted(seen_final_lr)} -- the LR curve is not being scaled consistently across ladder rungs")
    else:
        print(f"OK: final LR identical across the ladder: {seen_final_lr.pop():.6e}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, help="Path to a yaml config; resolves grid+fixed via generate_experiments() and uses the first generated experiment.")

    parser.add_argument("--ladder", default=None, help="Comma-separated subset sizes, e.g. 1000,5000,20000,50000")

    args = parser.parse_args()

    ladder = (

        [int(x) for x in args.ladder.split(",")]
        if args.ladder
        else DEFAULT_LADDER

    )

    if args.config:

        # Settings may live under grid:, so use the first generated
        # experiment. The schedule depends only on subset_size, which
        # is overridden per rung anyway.
        raw_config = load_yaml(args.config)

        config = generate_experiments(raw_config)[0]

    else:

        print("No --config given: using built-in defaults matching this repo's standard fixed: block.")

        for regime, max_steps in [("iso_epoch", None), ("iso_step", 938)]:

            config = dict(DEFAULT_CONFIG)
            config["training.regime"] = regime

            if max_steps is not None:
                config["training.max_steps"] = max_steps

            run_ladder(config, ladder)

        return

    run_ladder(config, ladder)


if __name__ == "__main__":
    main()
