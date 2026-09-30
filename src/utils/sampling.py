import numpy as np


# Fixed on purpose and independent of data_seed: every run, whatever
# its subset_size or seed, is scored on the same eval population.
EVAL_SPLIT_SEED = 20260101


def nested_subset(total_len, subset_size, seed):
    """Return the first `subset_size` indices of a seeded permutation.

    All subset sizes share one permutation per (total_len, seed), so
    subsets nest (1k within 5k within 20k): ladder rungs differ only
    in the amount of data, not in which examples were drawn.
    """
    permutation = np.random.default_rng(seed).permutation(total_len)

    n = (

        total_len
        if subset_size == "full"
        else min(int(subset_size), total_len)

    )

    return permutation[:n].tolist()


def fixed_eval_subset(total_len, eval_size):
    """Return fixed eval indices, independent of data_seed and size.

    Uses EVAL_SPLIT_SEED so every run is scored on the same examples.
    """
    permutation = np.random.default_rng(EVAL_SPLIT_SEED).permutation(total_len)

    n = (

        total_len
        if eval_size is None
        else min(int(eval_size), total_len)

    )

    return permutation[:n].tolist()


def nested_train_eval_split(total_len, subset_size, seed, eval_size=2000):
    """Reserve a fixed eval set first, then subsample the rest.

    Used for MNLI and SQuAD, whose eval set is carved out of the train
    split. Reserving eval before subsampling keeps it identical across
    every seed and ladder rung.
    """
    eval_indices = fixed_eval_subset(total_len, eval_size)

    remaining_mask = np.ones(total_len, dtype=bool)
    remaining_mask[eval_indices] = False
    remaining = np.nonzero(remaining_mask)[0]

    permutation = np.random.default_rng(seed).permutation(len(remaining))
    train_pool = remaining[permutation]

    n_train = (

        len(train_pool)
        if subset_size == "full"
        else min(int(subset_size), len(train_pool))

    )

    return train_pool[:n_train].tolist(), eval_indices
