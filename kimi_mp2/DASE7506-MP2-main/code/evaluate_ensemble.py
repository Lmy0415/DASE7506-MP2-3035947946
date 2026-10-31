import argparse
import json
from pathlib import Path

import numpy as np

from baselines import answer_groups, describe
from common import DATA, read_jsonl
from ranker import Ranker
from train_ranker import ROOT, cached_features, select, split_indices


def smooth_by_answer(bank, scores, strength):
    smoothed = scores.copy()
    for i, answers in enumerate(answer_groups(bank)):
        for answer in {answer for answer in answers if answer is not None}:
            members = [j for j, other in enumerate(answers) if other == answer]
            mean = scores[i, members].mean()
            smoothed[i, members] = (1 - strength) * scores[i, members] + strength * mean
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--feature-mode", choices=("base", "centered"), default="base")
    parser.add_argument("--output")
    args = parser.parse_args()
    models = [Ranker.load(ROOT / "runs" / "experiments" / f"{name}.npz") for name in args.names]
    train_bank, features = cached_features("train", args.feature_mode)
    validation_bank, validation_features = cached_features("validation", args.feature_mode)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    _, holdout, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout]
    holdout_references = [references[i] for i in holdout]
    internal_scores = [model.scores(features[holdout]) for model in models]
    validation_scores = [model.scores(validation_features) for model in models]
    result = {"models": args.names, "internal": {}, "validation": {}}
    combinations = {name: i for i, name in enumerate(args.names)}
    combinations["mean"] = None
    for name, index in combinations.items():
        if index is None:
            internal = np.mean(internal_scores, axis=0)
            validation = np.mean(validation_scores, axis=0)
        else:
            internal = internal_scores[index]
            validation = validation_scores[index]
        result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal))
        result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation))
        print(json.dumps({"model": name, "internal_hard": result["internal"][name]["hard_accuracy"], "validation_hard": result["validation"][name]["hard_accuracy"], "validation_all": result["validation"][name]["accuracy"]}))
    mean_internal = np.mean(internal_scores, axis=0)
    mean_validation = np.mean(validation_scores, axis=0)
    for strength in (0.25, 0.5, 0.75, 1.0):
        name = f"mean_answer_smoothing_{strength}"
        internal = smooth_by_answer(holdout_bank, mean_internal, strength)
        validation = smooth_by_answer(validation_bank, mean_validation, strength)
        result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal))
        result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation))
        print(json.dumps({"model": name, "internal_hard": result["internal"][name]["hard_accuracy"], "validation_hard": result["validation"][name]["hard_accuracy"], "validation_all": result["validation"][name]["accuracy"]}))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
