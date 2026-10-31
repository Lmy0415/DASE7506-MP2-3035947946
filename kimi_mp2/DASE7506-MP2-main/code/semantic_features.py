import re
from fractions import Fraction

import numpy as np

from common import parse_answer
from features import answer_value, numbers
from math_features import candidate_equations, normalize_expression

KEYWORDS = (
    "initial", "currently", "remaining", "left", "more", "less", "difference", "total",
    "each", "per", "twice", "double", "half", "percent", "split", "share", "average",
    "profit", "cost", "sell", "buy", "lost", "combined", "additional", "extra",
    "times", "rate", "speed", "minute", "hour", "day", "week", "second", "than",
)


def keyword_flags(text):
    lowered = text.lower()
    return [float(re.search(rf"\b{re.escape(keyword)}\b", lowered) is not None) for keyword in KEYWORDS]


def semantic_features(bank):
    rows = []
    for row in bank:
        question = row["question"]
        final_question = question.rsplit("?", 1)[0].rsplit(".", 1)[-1]
        question_flags = keyword_flags(question)
        final_question_flags = keyword_flags(final_question)
        question_numbers = set(numbers(question))
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        numeric_sets = [set(numbers(text)) for text in texts]
        group = []
        for i, text in enumerate(texts):
            tail = text[-700:]
            candidate_flags = keyword_flags(tail)
            equations = candidate_equations(text)
            answer_text = answers[i]
            answer = Fraction(answer_text) if answer_text is not None else None
            computed = [left for left, _, _, _ in equations]
            computed_valid = [left for left, _, valid, _ in equations if valid]
            computed_support = float(answer is not None and answer in computed)
            computed_valid_support = float(answer is not None and answer in computed_valid)
            own_intermediate = numeric_sets[i] - question_numbers
            if answer is not None:
                own_intermediate.discard(answer_value(answer_text))
            other_sets = [numeric_sets[j] - question_numbers for j in range(4) if j != i]
            shared_counts = [len(own_intermediate & other) for other in other_sets]
            normalized = normalize_expression(text)
            operation_counts = [normalized.count(symbol) for symbol in ("+", "-", "*", "/")]
            answer_magnitude = abs(answer_value(answer_text)) if answer is not None else 0.0
            question_magnitude = max([abs(value) for value in question_numbers] + [1.0])
            interactions = [q * c for q, c in zip(final_question_flags, candidate_flags)]
            group.append(
                question_flags + final_question_flags + candidate_flags + interactions + [
                    *[min(count, 20) / 20.0 for count in operation_counts],
                    min(len(computed), 20) / 20.0,
                    computed_support,
                    computed_valid_support,
                    float(answer is not None and answer in computed and not computed_valid_support),
                    np.mean(shared_counts) / 4.0,
                    max(shared_counts) / 4.0,
                    len(own_intermediate) / 20.0,
                    len(set.union(*other_sets) & own_intermediate) / max(len(own_intermediate), 1) if other_sets else 0.0,
                    np.log1p(answer_magnitude / question_magnitude) / 8.0,
                    len(set(numbers(tail)) & question_numbers) / max(len(question_numbers), 1),
                ]
            )
        rows.append(group)
    return np.asarray(rows, dtype=np.float32)
