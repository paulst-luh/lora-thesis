# Models

The model weights are not in git; only this README is. A config names its
model with `model.name`. A bare name without a `/` resolves to
`models/<name>` (`src/models/base_model.py`); a name with a `/` is used
as-is, as a path or a Hugging Face Hub id.

## Base model: Qwen2-7B

Every study config uses `model.name: qwen2-7b` (set in
`configs/_base.yaml`), i.e. `models/qwen2-7b`. Download it (about 15 GB)
from the repository root with the environment active:

```bash
hf download Qwen/Qwen2-7B --local-dir models/qwen2-7b
```

`python -m scripts.download_assets` does the same and also downloads the
datasets (see `data/README.md`). The model is public, so no token is
needed. The folder should then contain:

```
models/qwen2-7b/
  config.json
  generation_config.json
  model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors
  model.safetensors.index.json
  tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
```

## Other models

The model-comparison configs (`configs/outlook/`) and the smoke tests
(`configs/test/`) name other models in their `_model.yaml`, e.g.
`model.name: llama-3.1-8b-instruct`. Download each one into the folder of
that name:

```bash
hf download <hub-id> --local-dir models/<model.name>
```

Gated models (Llama, Gemma) require accepting the license on the model's
Hub page and a personal Hugging Face token. Set the token as an
environment variable (or log in with `hf auth login`), never in the code or
a config:

```bash
export HF_TOKEN=<your-huggingface-token>
```
