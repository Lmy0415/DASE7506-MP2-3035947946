import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices
from train_stacker import stack_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--leaf", type=int, default=30)
    args = parser.parse_args()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_parts = []
    validation_parts = []
    for filename in ("cv_math_h256_r2_e20_scores.npz", "cv_math_cosine50_r2_scores.npz"):
        with np.load(ROOT / "runs" / filename, allow_pickle=False) as saved:
            train_parts.append(stack_features(train_bank, features, saved["oof"], "base"))
            validation_parts.append(stack_features(validation_bank, validation_features, saved["validation"].mean(axis=0), "base"))
    train_x = np.concatenate(train_parts, axis=2)
    validation_x = np.concatenate(validation_parts, axis=2)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    rows = (4 * train_indices[:, None] + np.arange(4)).ravel()
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=args.leaf, l2_regularization=1.0, random_state=1729)
    model.fit(train_x.reshape(-1, train_x.shape[-1])[rows], labels.reshape(-1)[rows])
    internal_scores = model.predict_proba(train_x.reshape(-1, train_x.shape[-1]))[:, 1].reshape(-1, 4)[holdout_indices]
    validation_scores = model.predict_proba(validation_x.reshape(-1, validation_x.shape[-1]))[:, 1].reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump(model, destination / f"{args.name}.joblib")
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal": result["internal"]["hard_accuracy"], "validation": result["validation"]["hard_accuracy"]}))


if __name__ == "__main__":
    main()
