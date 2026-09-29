from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model, Qwen2ForQuestionAnswering
import torch
from torch import nn



class MyQwen2ForQuestionAnswering(Qwen2ForQuestionAnswering):
    base_model_prefix = "model"

def load_model(model_name, dataset_name, num_labels=None, pipeline="head", dtype="bfloat16"):

    # A bare name (no "/") resolves to the local models/<name> folder;
    # paths and HF hub ids are passed through unchanged.
    if "/" not in model_name:
        model_name = f"models/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if pipeline == "prompting":

        # AutoModelForCausalLM picks the class from the checkpoint's
        # config.json, so any causal LM works here. The LoRA module
        # names in lora_setup.py fit dense Llama-style models only, not
        # MoE checkpoints; get_peft_model() fails loudly on a mismatch.
        model = AutoModelForCausalLM.from_pretrained(

            model_name,

            dtype=getattr(torch, dtype),

            device_map="auto"

        )

    elif pipeline == "head":

        if dataset_name == "mnli":

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    dtype=torch.bfloat16,
                    device_map="auto"
                )
            )

        elif dataset_name == "squad":

            model = (
                MyQwen2ForQuestionAnswering
                .from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map="auto"
                )
            )

        elif dataset_name == "conll2003":

            model = (
                AutoModelForTokenClassification
                .from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    dtype=torch.bfloat16,
                    device_map="auto"
                )
            )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    return model, tokenizer