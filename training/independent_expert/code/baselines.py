import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, parse_answer, read_jsonl


def answer_groups(bank):
    return [[parse_answer(candidate["text"]) for candidate in row["candidates"]] for row in bank]


def correctness(bank, references):
    groups = answer_groups(bank)
    return np.array([
        [answer is not None and answer == parse_answer(reference["answer"]) for answer in answers]
        for answers, reference in zip(groups, references)
    ], dtype=bool)


def describe(bank, references, choices):
    labels = correctness(bank, references)
    answers = answer_groups(bank)
    chosen = np.asarray(choices)
    correct = labels[np.arange(len(bank)), chosen]
    distinct = np.array([len({answer for answer in row if answer is not None}) for row in answers])
    result = {
        "n": len(bank),
        "correct": int(correct.sum()),
        "accuracy": float(100 * correct.mean()),
        "hard_n": int((distinct > 1).sum()),
        "hard_correct": int(correct[distinct > 1].sum()),
        "hard_accuracy": float(100 * correct[distinct > 1].mean()),
        "easy_n": int((distinct == 1).sum()),
        "easy_correct": int(correct[distinct == 1].sum()),
        "patterns": {},
    }
    for row_index, row in enumerate(answers):
        counts = Counter(answer for answer in row if answer is not None)
        pattern = ":".join(map(str, sorted(counts.values(), reverse=True)))
        result["patterns"].setdefault(pattern, {"n": 0, "correct": 0})
        result["patterns"][pattern]["n"] += 1
        result["patterns"][pattern]["correct"] += int(correct[row_index])
    return result


def choose_majority(bank):
    choices = []
    for row in bank:
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        if not counts:
            choices.append(0)
            continue
        top_count = max(counts.values())
        winners = {answer for answer, count in counts.items() if count == top_count}
        choices.append(next((i for i, answer in enumerate(answers) if answer in winners), 0))
    return choices


def main():
    output = {}
    for split in ("train", "validation"):
        bank = read_jsonl(DATA / "candidates" / f"gsm8k_{split}.jsonl")
        references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
        methods = {
            "index_0": [0] * len(bank),
            "majority_answer": choose_majority(bank),
            "longest_trace": [max(range(4), key=lambda i: len(row["candidates"][i]["text"])) for row in bank],
            "shortest_trace": [min(range(4), key=lambda i: len(row["candidates"][i]["text"])) for row in bank],
        }
        output[split] = {name: describe(bank, references, choices) for name, choices in methods.items()}
    destination = Path(__file__).resolve().parents[1] / "runs" / "baselines.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
