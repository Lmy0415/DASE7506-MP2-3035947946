import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices
from train_stacker import stack_features


def instruction_weights(labels, indices):
    weights = np.ones((len(indices), 4), dtype=np.float64)
    for row, question_index in enumerate(indices):
        positive = labels[question_index].sum()
        negative = 4 - positive
        if positive and negative:
            weights[row, labels[question_index]] = 0.5 / positive
            weights[row, ~labels[question_index]] = 0.5 / negative
        else:
            weights[row] = 1.0 / 4
    return weights


def hard_negative_weights(labels, indices, scores, factor):
    weights = np.ones((len(indices), 4), dtype=np.float64)
    for row, question_index in enumerate(indices):
        wrong = np.flatnonzero(~labels[question_index])
        if len(wrong):
            hardest = wrong[np.argmax(scores[question_index, wrong])]
            weights[row, hardest] = factor
    return weights


def main():
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    with np.load(ROOT / "runs" / "cv_math_h256_r2_e20_scores.npz", allow_pickle=False) as saved:
        oof = saved["oof"]
        validation_neural = saved["validation"].mean(axis=0)
    train_x = stack_features(train_bank, features, oof, "base")
    validation_x = stack_features(validation_bank, validation_features, validation_neural, "base")
    train_indices, holdout_indices, _ = split_indices(train_bank)
    rows = (4 * train_indices[:, None] + np.arange(4)).ravel()
    x = train_x.reshape(-1, train_x.shape[-1])[rows]
    y = labels.reshape(-1)[rows]
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    destination = ROOT / "runs" / "experiments"
    results = {}
    models = {}

    def fit(name, weights, leaf=30):
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=leaf, l2_regularization=1.0, random_state=1729)
        model.fit(x, y, sample_weight=weights.reshape(-1))
        internal_scores = model.predict_proba(train_x.reshape(-1, train_x.shape[-1]))[:, 1].reshape(-1, 4)
        validation_scores = model.predict_proba(validation_x.reshape(-1, validation_x.shape[-1]))[:, 1].reshape(-1, 4)
        results[name] = {
            "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores[holdout_indices])),
            "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        }
        models[name] = (model, internal_scores, validation_scores)
        joblib.dump(model, destination / f"{name}.joblib")
        np.savez(destination / f"{name}_scores.npz", internal=internal_scores[holdout_indices], validation=validation_scores)
        print(json.dumps({"name": name, "internal": results[name]["internal"]["hard_accuracy"], "validation": results[name]["validation"]["hard_accuracy"]}), flush=True)

    fit("paper_baseline", np.ones((len(train_indices), 4)))
    balanced = instruction_weights(labels, train_indices)
    fit("paper_instruction_balanced", balanced)
    baseline_train_scores = models["paper_baseline"][1]
    for factor in (2.0, 4.0):
        hard = hard_negative_weights(labels, train_indices, baseline_train_scores, factor)
        fit(f"paper_hard_{int(factor)}", hard)
        fit(f"paper_balanced_hard_{int(factor)}", balanced * hard)
    second_scores = models["paper_hard_2"][1]
    hard_second = hard_negative_weights(labels, train_indices, second_scores, 2.0)
    fit("paper_hard_round3", hard_second)
    (destination / "paper_experiments.json").write_text(json.dumps({"results": results, "seconds": time.perf_counter() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
