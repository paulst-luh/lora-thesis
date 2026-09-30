# lora-thesis

Master's thesis: **Systematic Ablation Study of LoRA Hyperparameters.**
Fine-tunes Qwen2-7B with LoRA on four tasks (MNLI, SQuAD, CoNLL-2003,
GSM8K) and studies how rank, alpha, target modules, learning rate and
dataset size interact, compared against full fine-tuning (FFT) and a
zero-shot baseline.

## Setup

Requirements: Linux, Python 3.11 and one NVIDIA GPU with bf16 support and
at least 48 GB of memory (the runs peaked at 31 GB for LoRA and 44 GB for
FFT; the study used 80 GB A100s). About 20 GB of disk space for data and
model weights.

1. **Create the environment** in the repository root:

   ```bash
   python3.11 -m venv lora-env
   source lora-env/bin/activate
   pip install -r requirements.txt
   ```

   `torch` installs its default CUDA 12 build. For a CUDA 11.8 driver,
   install it first with
   `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118`.

2. **Download the datasets and the base model** (skips anything already
   present):

   ```bash
   python -m scripts.download_assets
   ```

   This writes `data/{mnli,squad,conll2003,gsm8k}` and `models/qwen2-7b`.
   All sources are public, so **no API key is needed**. Only the gated
   models of the model-comparison outlook (Llama, Gemma) require a Hugging
   Face token, set as an environment variable (never in the code):

   ```bash
   export HF_TOKEN=<your-huggingface-token>
   ```

3. **Check the setup** without a GPU:

   ```bash
   python -m scripts.dry_run_configs configs/mnli
   ```

## Running

A config expands into a grid of runs; pass an index to run a single one:

```bash
python -m src.experiments.runner configs/mnli/10_main_qv.yaml 0
```

On a SLURM cluster, every config has a job file (one array task per run),
and the wrappers in `slurm/scripts/` submit groups of them:

```bash
sbatch slurm/mnli/run_10_main_qv.slurm
bash slurm/scripts/run_mnli_prompting.sh
```

The job files assume the repository at `~/lora-thesis` with the
environment `lora-env` inside it, and the `kisski` partition; adjust
these for another cluster.

After the runs, build the result tables, figures and LaTeX output:

```bash
make figures                                # CSVs, all figures, tables
python -m scripts.export_hypershap_values   # HyperSHAP values
python -m scripts.hypershap_tables          # HyperSHAP LaTeX tables
```

## Folder structure

| Folder | Purpose |
|---|---|
| `configs/` | YAML experiment configs, one folder per dataset. `_base.yaml` holds settings shared by every run, `_task.yaml` the dataset settings, and each numbered file one experiment block (grid of runs). |
| `src/` | The Python code: training (`train.py`, `experiments/runner.py`), model loading and LoRA (`models/`), prompts, parsing and metrics (`datasets/`, `prompting/`), helpers (`utils/`) and the thesis figures and tables (`analysis/`). |
| `scripts/` | Standalone tools: downloading assets, dry runs, building the result CSVs, checking figures, HyperSHAP export and tables. |
| `slurm/` | SLURM job files per dataset and wrapper scripts that submit groups of jobs. |
| `docs/` | Design notes: experimental design, code architecture and figure plans. |
| `data/`, `models/` | Downloaded datasets and model weights (not in git). |
| `results/` | One JSON file per run plus logs, and the aggregated CSVs used for all analysis. |
| `figures/` | Generated figures |