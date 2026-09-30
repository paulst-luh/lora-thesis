"""Generate the single-seed cross-model outlook configs.

Writes configs/outlook/<model>/<dataset>/*.yaml for five other base
models, mirroring the main study's configs and scale (subset_size=5000,
epochs=3). Five blocks per model and dataset:

- 00_zeroshot: no training.
- 01_fft: full fine-tuning at lr 1e-5, the best FFT learning rate on
  three of four datasets (FFT diverges at LoRA-scale rates).
- 10/11/12_main_{qv,attention,full}: LoRA at lr 1e-4, the main
  sweep's best rate, with three (r, alpha) tiers from the main grid:
  low (4, 8), moderate (16, 32) and extreme (64, 128).

Seed 42 only: an outlook sample, not a replicated study.

Usage:
    python -m scripts.generate_outlook_configs
"""

import os


MODELS = [
    "gemma-4-e4b-it", "gpt-oss-20b", "deepseek-r1-distill-qwen-7b",
    "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3",
]
DATASETS = ["mnli", "squad", "conll2003", "gsm8k"]
TARGETS = ["qv", "attention", "full"]

# (tier, rank, alpha): all scaling ratio 2 and part of the main grid.
TIERS = [("low", 4, 8), ("moderate", 16, 32), ("extreme", 64, 128)]

FFT_LR = "1e-5"
LORA_LR = "1e-4"

OUT_ROOT = "configs/outlook"


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def model_yaml(model):
    return f"""#
# Model override for the {model} outlook probe -- included last (after
# the dataset _task.yaml) by every block config in this directory tree,
# so this is the only place model.name needs to change to point the
# whole probe at a different checkpoint. See models/{model}/ (downloaded
# via scripts/aggregate_other_models.py's sibling download script).
#

fixed:

  model.name: {model}
"""


def zeroshot_yaml(model, dataset):
    return f"""include:
  - ../../../{dataset}/_task.yaml
  - ../_model.yaml

experiment_group: {model}_outlook
experiment_name: {model}_outlook_{dataset}_00_zeroshot

grid:

  seed: [42]

fixed:

  output.root: results/outlook/{model}/{dataset}/00_zeroshot

  adaptation: none

  dataset.subset_size: 1
  training.learning_rate: 5e-05
  training.epochs: 1
"""


def fft_yaml(model, dataset):
    return f"""include:
  - ../../../{dataset}/_task.yaml
  - ../_model.yaml

experiment_group: {model}_outlook
experiment_name: {model}_outlook_{dataset}_01_fft

# {FFT_LR}, not the main-sweep LoRA winner (1e-4) -- see module docstring:
# FFT's own grid never included 1e-4, and this study's own FFT config
# comment warns it diverges at LoRA-scale learning rates.

grid:

  seed: [42]

fixed:

  output.root: results/outlook/{model}/{dataset}/01_fft

  adaptation: fft

  dataset.subset_size: 5000
  training.epochs: 3
  training.learning_rate: {FFT_LR}
"""


def lora_yaml(model, dataset, target, block_id):
    lora_config_lines = "\n".join(
        f"    - {{r: {r}, alpha: {alpha}}}      # {tier}" for tier, r, alpha in TIERS
    )
    return f"""include:
  - ../../../{dataset}/_task.yaml
  - ../_model.yaml

experiment_group: {model}_outlook
experiment_name: {model}_outlook_{dataset}_{block_id}_main_{target}

# 3 (rank, alpha) tiers, not the main study's full 8-pair grid -- low/
# moderate/extreme, all scaling_ratio 2, all points already tested in the
# main Qwen2-7B study (see module docstring). learning_rate fixed at the
# main-sweep's own winning LR across every dataset (fig_6_4e).

grid:

  lora_config:
{lora_config_lines}

  seed: [42]

fixed:

  output.root: results/outlook/{model}/{dataset}/{block_id}_main_{target}

  adaptation: lora
  lora.target: {target}

  dataset.subset_size: 5000
  training.epochs: 3
  training.learning_rate: {LORA_LR}
"""


def main():

    written = []

    for model in MODELS:

        _write(f"{OUT_ROOT}/{model}/_model.yaml", model_yaml(model))

        for dataset in DATASETS:

            base = f"{OUT_ROOT}/{model}/{dataset}"

            _write(f"{base}/00_zeroshot.yaml", zeroshot_yaml(model, dataset))
            _write(f"{base}/01_fft.yaml", fft_yaml(model, dataset))

            block_ids = {"qv": "10", "attention": "11", "full": "12"}
            for target in TARGETS:
                _write(
                    f"{base}/{block_ids[target]}_main_{target}.yaml",
                    lora_yaml(model, dataset, target, block_ids[target]),
                )

            written += [
                f"{base}/00_zeroshot.yaml", f"{base}/01_fft.yaml",
                f"{base}/10_main_qv.yaml", f"{base}/11_main_attention.yaml",
                f"{base}/12_main_full.yaml",
            ]

    n_runs_per_model_dataset = 1 + 1 + 3 * len(TARGETS)  # zeroshot + fft + 3 tiers x 3 targets
    total_runs = len(MODELS) * len(DATASETS) * n_runs_per_model_dataset

    print(f"wrote {len(written)} config files under {OUT_ROOT}/")
    print(
        f"{len(MODELS)} models x {len(DATASETS)} datasets x "
        f"{n_runs_per_model_dataset} runs/dataset = {total_runs} total runs represented"
    )


if __name__ == "__main__":
    main()
