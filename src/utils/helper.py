def get_output_dir(config):

    return config.get(
        "output.root",
        "results/debug"
    )

def get_trainable_stats(model):

    trainable = 0
    total = 0

    for p in model.parameters():

        num = p.numel()

        total += num

        if p.requires_grad:
            trainable += num

    pct = 100 * trainable / total

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": pct
    }
