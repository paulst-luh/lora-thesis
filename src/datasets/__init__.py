from .mnli import format_mnli, parse_mnli, MNLI_LABELS
from .gsm8k import format_gsm8k, parse_gsm8k
from .squad import format_squad, parse_squad
from .conll2003 import format_conll2003, parse_conll2003


FORMATTERS = {

    "mnli": format_mnli,

    "gsm8k": format_gsm8k,

    "squad": format_squad,

    "conll2003": format_conll2003,

}


PARSERS = {

    "mnli": parse_mnli,

    "gsm8k": parse_gsm8k,

    "squad": parse_squad,

    "conll2003": parse_conll2003,

}


# Label sets of the classification datasets. The labels are scored
# directly instead of being parsed from free generation.
VERBALIZERS = {

    "mnli": MNLI_LABELS,

}


def build_prompt_examples(
    dataset,
    dataset_name
):

    formatter = FORMATTERS[dataset_name]

    return dataset.map(

        formatter,

        remove_columns=dataset.column_names

    )
