import torch


@torch.no_grad()
def _choice_logprob(model, tokenizer, prompt, choice, max_length=512):

    # As in tokenize_prompt (prepare.py), the prompt's token count marks
    # where the choice begins in the full text.
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length
    )["input_ids"]

    prompt_length = len(prompt_ids)

    full = tokenizer(
        prompt + " " + choice,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)

    input_ids = full["input_ids"][0]

    if prompt_length >= len(input_ids):
        return float("-inf")

    logits = model(**full).logits[0]

    log_probs = torch.log_softmax(logits[:-1], dim=-1)

    choice_ids = input_ids[prompt_length:]

    choice_log_probs = log_probs[prompt_length - 1:].gather(
        -1,
        choice_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Mean rather than sum, so choices of different token length are
    # compared fairly.
    return choice_log_probs.mean().item()


@torch.no_grad()
def run_constrained_choice(
    model,
    tokenizer,
    dataset,
    choices,
    max_length=512
):

    model.eval()

    predictions = []

    for sample in dataset:

        scores = [

            _choice_logprob(
                model,
                tokenizer,
                sample["prompt"],
                choice,
                max_length
            )

            for choice in choices

        ]

        best = max(
            range(len(choices)),
            key=lambda i: scores[i]
        )

        predictions.append(choices[best])

    return predictions


@torch.no_grad()
def run_prompting(
    model,
    tokenizer,
    dataset,
    max_new_tokens=32,
    max_length=512,
    do_sample=False
):

    model.eval()

    predictions = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Greedy generate() output varied across identical runs, so prefer
    # deterministic kernels here (warn_only: ops without one still
    # run). The previous setting is restored on exit.
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:

        for sample in dataset:

            inputs = tokenizer(
                sample["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(model.device)

            outputs = model.generate(

                **inputs,

                max_new_tokens=max_new_tokens,

                do_sample=do_sample,

                temperature=None,

                top_p=None,

                pad_token_id=tokenizer.pad_token_id,

                eos_token_id=tokenizer.eos_token_id

            )

            generated = outputs[0][inputs["input_ids"].shape[1]:]

            prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            predictions.append(prediction)

    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=True)

    return predictions