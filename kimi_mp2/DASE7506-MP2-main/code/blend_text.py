import argparse
import json

import joblib
import numpy as np
from scipy import sparse

from baselines import describe
from common import DATA, read_jsonl
from ranker import Ranker
from train_ranker import ROOT, cached_features, select, split_indices
from train_text import documents


def normalize(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("neural")
    parser.add_argument("text")
    parser.add_argument("--output")
    args = parser.parse_args()
    neural = Ranker.load(ROOT / "runs" / "experiments" / f"{args.neural}.npz")
    bundle = joblib.load(ROOT / "runs" / "experiments" / f"{args.text}.joblib")
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    _, holdout, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout]
    holdout_references = [references[i] for i in holdout]
    train_text = bundle["vectorizer"].transform(documents(train_bank))
    validation_text = bundle["vectorizer"].transform(documents(validation_bank))
    if bundle["scaler"] is not None:
        train_hand = bundle["scaler"].transform(features[:, :, -33:].reshape(-1, 33)) * 0.1
        validation_hand = bundle["scaler"].transform(validation_features[:, :, -33:].reshape(-1, 33)) * 0.1
        train_text = sparse.hstack([train_text, sparse.csr_matrix(train_hand)], format="csr")
        validation_text = sparse.hstack([validation_text, sparse.csr_matrix(validation_hand)], format="csr")
    internal_neural = normalize(neural.scores(features[holdout]))
    validation_neural = normalize(neural.scores(validation_features))
    internal_text = normalize(bundle["model"].decision_function(train_text).reshape(-1, 4)[holdout])
    validation_text = normalize(bundle["model"].decision_function(validation_text).reshape(-1, 4))
    result = {"internal": {}, "validation": {}}
    for alpha in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        name = f"text_{alpha}"
        internal = (1 - alpha) * internal_neural + alpha * internal_text
        validation = (1 - alpha) * validation_neural + alpha * validation_text
        result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal))
        result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation))
        print(json.dumps({"alpha": alpha, "internal_hard": result["internal"][name]["hard_accuracy"], "validation_hard": result["validation"][name]["hard_accuracy"]}))
    if args.output:
        (ROOT / "runs" / args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
