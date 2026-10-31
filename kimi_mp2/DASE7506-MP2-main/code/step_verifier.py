import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from baselines import correctness, describe
from common import DATA, parse_answer, read_jsonl
from features import answer_value, numbers
from math_features import candidate_equations, normalize_expression
from train_ranker import ROOT, cached_features, select, split_indices


def step_rows(bank, features, labels=None):
    rows = []
    targets = []
    groups = []
    for question_index, row in enumerate(bank):
        positive_values = set()
        if labels is not None:
            for candidate_index in np.flatnonzero(labels[question_index]):
                for _, right, _, _ in candidate_equations(row["candidates"][candidate_index]["text"]):
                    positive_values.add(right)
        question_numbers = set(numbers(row["question"]))
        group = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            equations = candidate_equations(text)
            answer = parse_answer(text)
            answer = float(answer_value(answer)) if answer is not None else None
            result_chains = [
                {step[1] for step in candidate_equations(other["text"])}
                for other in row["candidates"]
            ]
            steps = []
            bad_seen = False
            for step_index, (left, right, valid, line) in enumerate(equations):
                if labels is not None:
                    if labels[question_index, candidate_index]:
                        target = 1
                    else:
                        target = int(not bad_seen and right in positive_values)
                        if right not in positive_values:
                            bad_seen = True
                    targets.append(target)
                right_float = float(right)
                left_float = float(left)
                support = sum(right in chain for chain in result_chains) - 1
                normalized = normalize_expression(line)
                custom = [
                    step_index / max(len(equations) - 1, 1),
                    min(len(equations), 15) / 15.0,
                    float(valid),
                    np.sign(left_float),
                    np.log1p(abs(left_float)) / 12.0,
                    np.sign(right_float),
                    np.log1p(abs(right_float)) / 12.0,
                    float(right_float.is_integer()),
                    float(answer is not None and right_float == answer),
                    float(right_float in question_numbers),
                    support / 3.0,
                    min(len(line), 500) / 500.0,
                    min(normalized.count("+"), 10) / 10.0,
                    min(normalized.count("-"), 10) / 10.0,
                    min(normalized.count("*"), 10) / 10.0,
                    min(normalized.count("/"), 10) / 10.0,
                ]
                rows.append(np.concatenate([features[question_index, candidate_index, 1024:], custom]))
                steps.append(len(rows) - 1)
            group.append(steps)
        groups.append(group)
    return np.asarray(rows, dtype=np.float32), np.asarray(targets), groups


def aggregate(scores, groups, n):
    output = np.zeros((n, 4, 3))
    for question_index, group in enumerate(groups):
        for candidate_index, indices in enumerate(group):
            if not indices:
                output[question_index, candidate_index] = [0.5, 0.5, 0.0]
                continue
            values = np.clip(scores[indices], 1e-6, 1)
            output[question_index, candidate_index] = [values.min(), values.mean(), np.log(values).mean()]
    return output


def main():
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    rows, targets, groups = step_rows(train_bank, features, labels)
    validation_rows, _, validation_groups = step_rows(validation_bank, validation_features)
    selected = []
    for question_index in train_indices:
        for candidate_indices in groups[question_index]:
            selected.extend(candidate_indices)
    selected = np.asarray(selected)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=50, l2_regularization=1.0, random_state=1729)
    positive = targets[selected].sum()
    negative = len(selected) - positive
    weights = np.where(targets[selected] == 1, len(selected) / (2 * positive), len(selected) / (2 * negative))
    model.fit(rows[selected], targets[selected], sample_weight=weights)
    step_scores = model.predict_proba(rows)[:, 1]
    validation_step_scores = model.predict_proba(validation_rows)[:, 1]
    internal = aggregate(step_scores, groups, len(train_bank))[holdout_indices]
    validation = aggregate(validation_step_scores, validation_groups, len(validation_bank))
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {"steps": len(rows), "positive_rate": float(targets[selected].mean()), "internal": {}, "validation": {}}
    for index, name in enumerate(("min", "mean", "logmean")):
        result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal[:, :, index]))
        result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation[:, :, index]))
        print(json.dumps({"aggregation": name, "internal": result["internal"][name]["hard_accuracy"], "validation": result["validation"][name]["hard_accuracy"]}))
    destination = ROOT / "runs" / "experiments"
    joblib.dump(model, destination / "step_verifier.joblib")
    np.savez(destination / "step_verifier_scores.npz", internal=internal, validation=validation)
    result["seconds"] = time.perf_counter() - started
    (destination / "step_verifier.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
