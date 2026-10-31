import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices


def aggregate(bank, scores, mode):
    adjusted = scores.copy()
    for i, answers in enumerate(answer_groups(bank)):
        for answer in {answer for answer in answers if answer is not None}:
            members = [j for j, other in enumerate(answers) if other == answer]
            if mode == "sum":
                adjusted[i, members] = scores[i, members].sum()
            elif mode == "logsum":
                adjusted[i, members] = np.log(np.clip(scores[i, members], 1e-8, 1)).sum()
            elif mode == "mean":
                adjusted[i, members] = scores[i, members].mean()
    return adjusted


def main():
    train_bank, _ = cached_features("train", "base")
    validation_bank, _ = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    _, holdout, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout]
    holdout_labels = correctness(holdout_bank, holdout_references := [references[i] for i in holdout])
    validation_labels = correctness(validation_bank, validation_references)
    saved = [np.load(ROOT / "runs" / "experiments" / f"{name}_scores.npz") for name in ("stack_hgb_all", "stack_hgb_leaf10")]
    internal_scores = np.mean([item["internal"] for item in saved], axis=0)
    validation_scores = np.mean([item["validation"] for item in saved], axis=0)
    hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(holdout_bank)])
    rng = np.random.default_rng(42)
    calibration = []
    for mask in (hard, ~hard):
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        calibration.extend(indices[:len(indices) // 2])
    calibration = np.array(sorted(calibration))
    evaluation = np.setdiff1d(np.arange(len(holdout)), calibration)
    result = {}
    for method in ("platt", "isotonic"):
        if method == "platt":
            model = LogisticRegression(C=1.0, max_iter=1000)
            model.fit(internal_scores[calibration].reshape(-1, 1), holdout_labels[calibration].reshape(-1))
            transform = lambda values: model.predict_proba(values.reshape(-1, 1))[:, 1].reshape(values.shape)
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            model.fit(internal_scores[calibration].reshape(-1), holdout_labels[calibration].reshape(-1))
            transform = lambda values: model.predict(values.reshape(-1)).reshape(values.shape)
        calibrated_internal = transform(internal_scores)
        calibrated_validation = transform(validation_scores)
        for mode in ("single", "mean", "sum", "logsum"):
            internal = calibrated_internal if mode == "single" else aggregate(holdout_bank, calibrated_internal, mode)
            validation = calibrated_validation if mode == "single" else aggregate(validation_bank, calibrated_validation, mode)
            eval_bank = [holdout_bank[i] for i in evaluation]
            eval_references = [holdout_references[i] for i in evaluation]
            name = f"{method}_{mode}"
            result[name] = {
                "internal_holdout": describe(eval_bank, eval_references, select(eval_bank, internal[evaluation])),
                "validation": describe(validation_bank, validation_references, select(validation_bank, validation)),
            }
            print(json.dumps({"name": name, "internal": result[name]["internal_holdout"]["hard_accuracy"], "validation": result[name]["validation"]["hard_accuracy"]}))
    (ROOT / "runs" / "experiments" / "calibrated_voting.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
