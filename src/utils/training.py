from transformers import TrainingArguments
import numpy as np
import os
import random
import torch

from src.utils.helper import get_output_dir

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_schedule(config):
    """Return (max_steps, num_train_epochs) for the training regime.

    - iso_epoch (default): fixed epoch count, so optimizer steps scale
      with subset_size. An explicit training.max_steps is still
      honored as a hard cap.
    - iso_step: fixed max_steps on every ladder rung. Trainer ignores
      num_train_epochs when max_steps > 0, so it is set to 1.
    """
    regime = config.get("training.regime", "iso_epoch")

    if regime == "iso_epoch":

        return config.get("training.max_steps", -1), config["training.epochs"]

    elif regime == "iso_step":

        return config["training.max_steps"], 1

    else:

        raise ValueError(
            f"Unknown training.regime: {regime}"
        )


def build_training_args(config,seed):

    output_root = get_output_dir(config)

    tmp_dir = os.path.join(output_root, "tmp")

    os.makedirs(tmp_dir, exist_ok=True)

    max_steps, num_train_epochs = resolve_schedule(config)

    return TrainingArguments(

        output_dir=tmp_dir,

        optim=config.get("training.optimizer"),

        weight_decay=config.get("training.weight_decay"),

        learning_rate=float(config["training.learning_rate"]),

        # transformers>=5 dropped warmup_ratio; a float in [0, 1) passed
        # as warmup_steps is read as a ratio of the total steps.
        warmup_steps=config.get("training.warmup_ratio", 0.03),
        lr_scheduler_type=config.get("training.lr_scheduler_type"),

        per_device_train_batch_size=config.get("training.batch_size"),

        per_device_eval_batch_size=config.get("training.batch_size"),

        gradient_accumulation_steps=config.get("training.grad_accum"),

        bf16=True,
        gradient_checkpointing=True,

        max_steps=max_steps,

        num_train_epochs=num_train_epochs,

        # Prompting runs are evaluated by generation after training,
        # not by Trainer's eval loop.
        eval_strategy=("epoch" if config.get("pipeline.type") == "head" else "no"),
        save_strategy="no",

        logging_steps=20,

        seed=seed,
        data_seed=seed,

        report_to="none",

        remove_unused_columns=False
    )