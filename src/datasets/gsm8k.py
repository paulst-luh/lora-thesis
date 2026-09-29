import re

from sklearn.metrics import accuracy_score


PROMPT_TEMPLATE = """
Solve the following math word problem. Show your reasoning step by step, then give the final answer after "####".

Problem:
{question}

Solution:
"""


# Prompting pipeline only: GSM8K is generative (chain of thought), so
# there is no head variant.

def format_gsm8k(example):

    return {

        "prompt": PROMPT_TEMPLATE.format(

            question=example["question"]

        ),

        "target": example["answer"],

        "label": None,

        "metadata": {}

    }


def parse_gsm8k(text):

    # Take the number after the GSM8K "####" answer marker, otherwise
    # the last number in the text.
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)

    if match:
        number = match.group(1)

    else:

        numbers = re.findall(r"-?[\d,]+\.?\d*", text)

        if not numbers:
            return None

        number = numbers[-1]

    return number.replace(",", "").rstrip(".")


def compute_gsm8k_metrics(predictions, references):

    return {

        "eval_accuracy": accuracy_score(
            references,
            predictions
        )

    }
