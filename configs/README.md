# Configs

Every run is defined by a YAML config in this folder; nothing is hardcoded.

## Layout

| Path | Purpose |
|---|---|
| `_base.yaml` | Settings shared by every run: model, optimizer, batch size, LR schedule, data seed, evaluation size. |
| `<dataset>/_task.yaml` | Dataset settings (task type, sequence length, generation length). Includes `_base.yaml`. |
| `<dataset>/NN_*.yaml` | One experiment block per file: `00` zero-shot, `01` FFT, `10`-`12` main sweep (Query-Value / Attention / Full LoRA), `20`-`23` dataset-size ladders. A file with a suffix (e.g. `10_main_qv_ratio1_extend.yaml`) is a follow-up to the block it is named after. |
| `<dataset>/{qv,attention,full,fft}.yaml` | Earlier classification-head pipeline (`pipeline.type: head`); not part of the study. |
| `templates/` | Fill-in templates for a zero-shot, FFT or LoRA block. |
| `outlook/`, `test/` | Model comparison and small-model smoke tests; each model folder has a `_model.yaml` that sets `model.name`. |
| `tasks/` | Shared task overrides (GSM8K reasoning lengths). |

## How a config expands into runs

```yaml
include: _task.yaml          # inherits _task.yaml, which inherits _base.yaml

experiment_group: mnli_prompting
experiment_name: mnli_prompting_main_qv

grid:                        # every combination is one run
  lora_config:               # (r, alpha) pairs, expanded together, not crossed
    - {r: 8, alpha: 8}
    - {r: 8, alpha: 16}
  training.learning_rate: [1e-4, 3e-4, 5e-4]
  seed: [42, 43, 44]

fixed:                       # the same value for every run
  output.root: results/mnli/prompting_main_qv
  adaptation: lora
  lora.target: qv
  dataset.subset_size: 5000
  training.epochs: 3
```

- `fixed:` blocks are merged along the `include:` chain; later layers win,
  so any `_base.yaml` or `_task.yaml` key can be overridden here.
- The number of runs is the product of the grid list lengths (here
  2 x 3 x 3 = 18). A run is addressed by its index `0` to `N-1`.
- Keys use dot notation (`training.learning_rate`).

## Creating your own config

1. Copy a template into a dataset folder (`include: _task.yaml` resolves
   relative to the file):

   ```bash
   cp configs/templates/_template_lora.yaml configs/mnli/30_my_block.yaml
   ```

2. Fill in the placeholders. The keys you usually set:

   | Key | Values |
   |---|---|
   | `adaptation` | `lora`, `fft` or `none` (zero-shot) |
   | `lora.target` | `qv` (q, v), `attention` (q, k, v, o), `full` (attention + MLP) |
   | `lora_config` | list of `{r: ..., alpha: ...}` pairs (grid only) |
   | `training.learning_rate` | the study used 1e-4, 3e-4, 5e-4 for LoRA and 1e-5, 3e-5, 5e-5 for FFT |
   | `seed` | training seed; the study used 42, 43, 44 |
   | `dataset.subset_size` | number of training examples |
   | `training.epochs` | number of epochs |
   | `training.regime`, `training.max_steps` | `iso_step` fixes the optimizer steps instead of the epochs |
   | `output.root` | where the run JSONs and logs are written |

3. Check the expansion without a GPU:

   ```bash
   python -m scripts.dry_run_configs configs/mnli/30_my_block.yaml
   ```

4. Run a single index, or all of them on SLURM by copying a job file
   (e.g. `slurm/mnli/run_10_main_qv.slurm`) and changing the job name,
   `--array=0-<N-1>%4`, the log and output paths, and the config path:

   ```bash
   python -m src.experiments.runner configs/mnli/30_my_block.yaml 0
   ```

## Rules

- Keep `experiment_group: <dataset>_prompting` and
  `experiment_name: <dataset>_prompting_<block>` to have the runs included
  in `results/prompting_study_final.csv`; the part after the prefix becomes
  the `block` column. Use a different group to keep a trial run out.
- Change one factor at a time relative to an existing block. If two
  factors change together, their effects cannot be separated.
- Leave `dataset.data_seed` and `dataset.eval_size` at their base values:
  they fix which examples are trained and evaluated on across all runs.
