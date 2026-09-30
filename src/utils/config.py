import yaml
import itertools
import os

def load_yaml(path):
    """Load a YAML config and resolve its `include:` chain.

    `include:` names one file or a list of files, relative to this
    file's directory. Their `fixed:` blocks are merged in order and
    this file's own `fixed:` wins. `grid:` is never merged.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    include = config.pop("include", None)

    if include is not None:

        includes = [include] if isinstance(include, str) else include

        merged_fixed = {}

        for included_path in includes:

            base = load_yaml(os.path.join(os.path.dirname(path), included_path))

            merged_fixed.update(base.get("fixed", {}))

        config["fixed"] = {
            **merged_fixed,
            **config.get("fixed", {})
        }

    return config


def generate_experiments(config):
    """Expand the config's grid into one config dict per run.

    Args:
        config (dict): Experiment config with 'grid' and 'fixed'.

    Returns:
        list[dict]: One config per grid combination.
    """
    if "grid" not in config:

        raise ValueError(
            "Config missing grid section"
        )

    grid = config["grid"]

    # Paired (r, alpha) entries are expanded together, not crossed.
    if "lora_config" in grid:

        lora_configs = grid["lora_config"]

        grid = {
            k: v
            for k, v in grid.items()
            if k != "lora_config"
        }

        keys = list(grid.keys())
        values = list(grid.values())

        combinations = []

        for lora_cfg in lora_configs:

            for combo in itertools.product(*values):

                combinations.append(
                    (
                        lora_cfg,
                        *combo
                    )
                )

        keys = ["lora_config"] + keys

    else:

        keys = list(grid.keys())

        values = list(grid.values())

        combinations = itertools.product(*values)

    experiments = []

    for combo in combinations:

        exp = {}

        exp["experiment_name"] = config.get(
            "experiment_name",
            "unnamed"
        )

        exp["experiment_group"] = config.get(
            "experiment_group",
            "default"
        )

        exp.update(
            config.get("fixed", {})
        )

        # Grid values override fixed values of the same key.
        combo_dict = dict(
            zip(keys, combo)
        )

        if "lora_config" in combo_dict:

            exp["lora.r"] = combo_dict["lora_config"]["r"]

            exp["lora.alpha"] = combo_dict["lora_config"]["alpha"]

            del combo_dict["lora_config"]

        exp.update(combo_dict)

        experiments.append(exp)

    return experiments