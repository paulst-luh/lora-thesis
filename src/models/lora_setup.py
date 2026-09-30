from peft import LoraConfig, get_peft_model


# Module names shared by Llama-style models (Qwen, Llama, Gemma and
# dense DeepSeek checkpoints).
LORA_TARGETS = {

    # Classic LoRA baseline.
    "qv": [
        "q_proj",
        "v_proj"
    ],

    # Attention-only adaptation.
    "attention": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    # All attention and MLP projections.
    "full": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj"
    ]
}


# GPT-2 fuses Q/K/V into one c_attn matrix and uses "c_proj" for both
# the attention output and the MLP down projection, so the parent name
# is added to tell them apart. There is no true "qv" target: c_attn
# also adapts K. It is a smoke-test stand-in only and must not be
# compared with other models' qv runs.
LORA_TARGETS_GPT2 = {

    "qv": [
        "c_attn"
    ],

    "attention": [
        "c_attn",
        "attn.c_proj"
    ],

    "full": [
        "c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj"
    ]
}


def apply_lora(model, config):

    target_key = config.get(
        "lora.target",
        "attention"
    )

    target_table = (
        LORA_TARGETS_GPT2
        if model.config.model_type == "gpt2"
        else LORA_TARGETS
    )

    if target_key not in target_table:
        raise ValueError(
            f"Unknown lora.target: {target_key}"
        )

    targets = target_table[target_key]

    dataset_name = config["dataset.name"]

    pipeline = config.get("pipeline.type", "head")

    if pipeline == "prompting":
        peft_task = "CAUSAL_LM"

    elif dataset_name == "mnli":
        peft_task = "SEQ_CLS"

    elif dataset_name == "squad":
        peft_task = "QUESTION_ANS"

    elif dataset_name == "conll2003":
        peft_task = "TOKEN_CLS"

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    print(f"LoRA target = {target_key}")
    print(f"Modules = {targets}")
    print(f"Dataset = {dataset_name}")

    peft_config = LoraConfig(
        r=config["lora.r"],
        lora_alpha=config["lora.alpha"],
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type=peft_task
    )

    model = get_peft_model(
        model,
        peft_config
    )

    model.print_trainable_parameters()

    return model