import json
import os
import time

from src.utils.helper import get_output_dir


# The headline metric key differs by task and pipeline (Trainer adds
# an "eval_" prefix, gsm8k has no F1). Each run has exactly one.
METRIC_PRIORITY = [
    "eval_f1",
    "f1",
    "eval_accuracy",
    "accuracy"
]


def _unique_id():
    # Nanoseconds plus PID: with 1-second resolution, concurrent SLURM
    # array tasks silently overwrote each other's files.
    return f"{time.time_ns()}_{os.getpid()}"


def _select_metric(metrics):

    for key in METRIC_PRIORITY:

        if key in metrics:
            return metrics[key]

    return None


def _compute_regime(subset_size):

    # Data-size bucket for RQ1. The 10000 cut-off lies between the
    # 5000 and 20000 subset sizes.
    if subset_size is None:
        return None

    if subset_size == "full":
        return "full"

    return "low" if int(subset_size) < 10000 else "high"


def build_flat_row(config, metrics, trainable_stats):
    """Return one flat, pandas-ready row with every grouping factor.

    Every key is always written. A value is None (NaN in pandas) only
    when the factor does not apply, e.g. rank and alpha for FFT runs.
    """
    rank = config.get("lora.r")
    alpha = config.get("lora.alpha")

    scaling_ratio = (
        alpha / rank
        if rank not in (None, 0) and alpha is not None
        else None
    )

    return {

        "dataset": config.get("dataset.name"),

        "pipeline": config.get("pipeline.type"),

        "adaptation": config.get("adaptation"),

        "lora_target": config.get("lora.target"),

        "rank": rank,

        "alpha": alpha,

        "scaling_ratio": scaling_ratio,

        "learning_rate": config.get("training.learning_rate"),

        "subset_size": config.get("dataset.subset_size"),

        "regime": _compute_regime(config.get("dataset.subset_size")),

        # Training schedule (iso_epoch/iso_step), not to be confused
        # with the data-size "regime" above.
        "training_regime": config.get("training.regime", "iso_epoch"),

        "seed": config.get("seed"),

        "metric": _select_metric(metrics),

        "parse_failure_rate": metrics.get("parse_failure_rate"),

        "runtime_s": metrics.get("runtime_seconds"),

        "peak_mem_gb": metrics.get("peak_allocated_vram_gb"),

        "trainable_params": trainable_stats.get("trainable_params"),

        "optimizer_steps": metrics.get("optimizer_steps")

    }


def save_results(
    config,
    metrics,
    trainable_stats,
    sample_predictions,
    confusion_matrix_data
):

    output_root = get_output_dir(config)

    metrics_dir = os.path.join(
        output_root,
        "metrics"
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )

    filename = os.path.join(
        metrics_dir,
        f"result_{_unique_id()}.json"
    )

    with open(
        filename,
        "w"
    ) as f:

        json.dump(
            {

                "experiment_name":
                    config.get("experiment_name","unnamed"),

                "experiment_group":
                    config.get("experiment_group","unknown"),

                "method":
                    config.get("adaptation"),

                "row":
                    build_flat_row(config, metrics, trainable_stats),

                "config":
                    config,

                "controlled_variables": {

                    "lora_r":
                        config.get("lora.r"),

                    "lora_alpha":
                        config.get("lora.alpha"),

                    "learning_rate":
                        config.get("training.learning_rate"),

                    "seed":
                        config.get("seed")

                },

                "fixed_variables": {

                    "model":
                        config.get("model.name"),
                    "model_dtype":
                        config.get("model.dtype", "bfloat16"),
                    "dataset":
                        config.get("dataset.name"),

                    "data_seed":
                        config.get("dataset.data_seed"),

                    "task":
                        config.get("task.name"),

                    "lora_target":
                        config.get("lora.target","baseline"),
    
                    "max_steps":
                        config.get("training.max_steps", -1),

                    "epochs":
                        config.get("training.epochs"),

                    "batch_size":
                        config.get("training.batch_size"),

                    "grad_accum":
                        config.get("training.grad_accum"),

                    "weight_decay":
                        config.get("training.weight_decay"),

                    "warmup_ratio":
                        config.get("training.warmup_ratio"),

                    "loss_on_completion":
                        config.get("training.loss_on_completion"),

                    "optimizer":
                        config.get("training.optimizer"),

                    "scheduler":
                        config.get("training.lr_scheduler_type")

                },

                "preprocessing": {

                    "tokenizer":
                        config.get("model.name"),

                    "max_length":
                        config.get("dataset.max_seq_length", 512),

                    "truncation":
                        True,

                    "shuffle":
                        True,

                    "train_size":
                        0.8,

                    "test_size":
                        0.2,

                    "subset_size":
                        config.get("dataset.subset_size")
                },


                "trainable":
                    trainable_stats,

                "metrics":
                    metrics,

                "confusion_matrix":
                    confusion_matrix_data,
                
                "sample_predictions":
                    sample_predictions

            },
            f,
            indent=2
        )

    print(
        f"Saved {filename}"
    )

def save_loss_history(
    config,
    loss_history
):

    output_root = get_output_dir(config)

    loss_dir = os.path.join(
        output_root,
        "loss_curves"
    )

    os.makedirs(
        loss_dir,
        exist_ok=True
    )

    filename = os.path.join(
        loss_dir,
        f"loss_{_unique_id()}.json"
    )

    with open(filename, "w") as f:

        json.dump(
            loss_history,
            f,
            indent=2
        )