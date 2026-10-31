from fractions import Fraction

import numpy as np

from common import parse_answer
from features import numbers

CONSTANTS = (Fraction(1), Fraction(2), Fraction(12), Fraction(60), Fraction(100))


def combine(left, right):
    operations = [(left + right, 0), (left - right, 1), (left * right, 2)]
    if right != 0:
        operations.append((left / right, 3))
    if left != 0:
        operations.append((right / left, 5))
    return operations


def better(signature, existing):
    if existing is None:
        return True
    return (signature[0].bit_count(), -len(signature[2])) > (existing[0].bit_count(), -len(existing[2]))


def reachability_features(bank):
    output = []
    for row in bank:
        raw_numbers = list(dict.fromkeys(numbers(row["question"])))
        slots = [Fraction(str(value)) for value in raw_numbers[:8]]
        levels = []
        initial = {}
        for index, value in enumerate(slots):
            initial[value] = (1 << index, (0, 0, 0, 0, 0, 0), ())
        for index, value in enumerate(CONSTANTS):
            signature = (0, (0, 0, 0, 0, 0, 0), (index,))
            if better(signature, initial.get(value)):
                initial[value] = signature
        levels.append(initial)
        for level in (1, 2):
            current = {}
            previous = levels[level - 1]
            for left_value, left_signature in previous.items():
                for right_value, right_signature in initial.items():
                    for value, operation in combine(left_value, right_value):
                        if abs(value) > 10 ** 12:
                            continue
                        operations = list(left_signature[1])
                        operations[operation] += 1
                        signature = (
                            left_signature[0] | right_signature[0],
                            tuple(operations),
                            left_signature[2] + right_signature[2],
                        )
                        if better(signature, current.get(value)):
                            current[value] = signature
            levels.append(current)
        group = []
        for candidate in row["candidates"]:
            answer = parse_answer(candidate["text"])
            value = Fraction(answer) if answer is not None else None
            matches = []
            if value is not None:
                for level, states in enumerate(levels):
                    if value in states:
                        matches.append((level, states[value]))
            if not matches:
                group.append([0.0] * 14)
                continue
            matches.sort(key=lambda item: (-item[1][0].bit_count(), len(item[1][2]), item[0]))
            level, signature = matches[0]
            coverage = signature[0].bit_count() / max(len(slots), 1)
            constant_flags = [float(index in signature[2]) for index in (2, 3, 4)]
            group.append([
                1.0,
                level / 2.0,
                coverage,
                float(signature[0].bit_count() == len(slots)),
                min(len(matches), 3) / 3.0,
                *[min(count, 3) / 3.0 for count in signature[1]],
                *constant_flags,
            ])
        output.append(group)
    return np.asarray(output, dtype=np.float32)
