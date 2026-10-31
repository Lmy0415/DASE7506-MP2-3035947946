import argparse
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from features import numbers
from math_features import candidate_equations
from train_ranker import ROOT, cached_features, select, split_indices


def signatures(bank):
    rows = []
    for row in bank:
        question_numbers = list(dict.fromkeys(numbers(row["question"])))[:12]
        group = []
        for candidate in row["candidates"]:
            text = candidate["text"]
            equation_text = "\n".join(line for line in text.splitlines() if "=" in line)
            used = numbers(equation_text)
            slots = [min(used.count(value), 4) / 4.0 for value in question_numbers]
            slots += [0.0] * (12 - len(slots))
            equations = candidate_equations(text)
            valid = sum(equation[2] for equation in equations)
            operations = [min(equation_text.count(symbol), 20) / 20.0 for symbol in ("+", "-", "*", "/")]
            group.append(slots + operations + [min(len(equations), 20) / 20.0, valid / max(len(equations), 1), len(set(used)) / 20.0])
        rows.append(group)
    return np.asarray(rows, dtype=np.float32)


def retrieve_scores(train_questions, train_signatures, train_labels, query_questions, query_signatures,
                    vectorizer, k=30, sigma=0.5):
    train_x = vectorizer.transform(train_questions)
    query_x = vectorizer.transform(query_questions)
    output = np.zeros((len(query_questions), 4))
    for start in range(0, len(query_questions), 128):
        similarities = (query_x[start:start + 128] @ train_x.T).toarray()
        for local, query_index in enumerate(range(start, min(start + 128, len(query_questions)))):
            nearest = np.argpartition(similarities[local], -k)[-k:]
            question_weights = np.maximum(similarities[local, nearest], 0)
            question_weights /= max(question_weights.sum(), 1e-8)
            for candidate in range(4):
                differences = train_signatures[nearest] - query_signatures[query_index, candidate]
                distances = np.square(differences).sum(axis=2)
                similarities_to_programs = np.exp(-distances / (2 * sigma * sigma))
                evidence = []
                for row_index, neighbor in enumerate(nearest):
                    correct = similarities_to_programs[row_index, train_labels[neighbor]]
                    wrong = similarities_to_programs[row_index, ~train_labels[neighbor]]
                    correct_max = correct.max() if len(correct) else 0.0
                    wrong_max = wrong.max() if len(wrong) else 0.0
                    evidence.append(correct_max - wrong_max)
                output[query_index, candidate] = np.dot(question_weights, evidence)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.5)
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, _ = cached_features("train", "base")
    validation_bank, _ = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(train_bank)])
    train_indices = train_indices[hard[train_indices]]
    train_signatures = signatures(train_bank)
    validation_signatures = signatures(validation_bank)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000, stop_words="english")
    vectorizer.fit([train_bank[i]["question"] for i in train_indices])
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    internal_scores = retrieve_scores(
        [train_bank[i]["question"] for i in train_indices], train_signatures[train_indices], labels[train_indices],
        [row["question"] for row in holdout_bank], train_signatures[holdout_indices], vectorizer, args.k, args.sigma)
    validation_scores = retrieve_scores(
        [train_bank[i]["question"] for i in train_indices], train_signatures[train_indices], labels[train_indices],
        [row["question"] for row in validation_bank], validation_signatures, vectorizer, args.k, args.sigma)
    result = {
        "config": vars(args),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
