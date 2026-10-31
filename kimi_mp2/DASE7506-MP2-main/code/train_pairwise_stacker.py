import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices
from train_stacker import stack_features


def training_pairs(features, labels, indices):
    rows = []
    for i in indices:
        for correct in np.flatnonzero(labels[i]):
            for wrong in np.flatnonzero(~labels[i]):
                rows.append(features[i, correct] - features[i, wrong])
    positive = np.asarray(rows, dtype=np.float32)
    return np.concatenate([positive, -positive]), np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])


def pairwise_scores(model, features):
    n = len(features)
    differences = []
    for i in range(4):
        for j in range(i + 1, 4):
            differences.append(features[:, i] - features[:, j])
    probabilities = model.predict_proba(np.concatenate(differences))[:, 1].reshape(6, n)
    scores = np.zeros((n, 4))
    pair = 0
    for i in range(4):
        for j in range(i + 1, 4):
            scores[:, i] += probabilities[pair]
            scores[:, j] += 1 - probabilities[pair]
            pair += 1
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--leaf", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    with np.load(ROOT / "runs" / "cv_math_h256_r2_e20_scores.npz", allow_pickle=False) as saved:
        oof = saved["oof"]
        validation_neural = saved["validation"].mean(axis=0)
    train_x = stack_features(train_bank, features, oof)
    validation_x = stack_features(validation_bank, validation_features, validation_neural)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    pair_x, pair_y = training_pairs(train_x, labels, train_indices)
    model = HistGradientBoostingClassifier(max_iter=args.iterations, learning_rate=0.05,
                                           max_leaf_nodes=args.nodes, min_samples_leaf=args.leaf,
                                           l2_regularization=1.0, random_state=1729)
    model.fit(pair_x, pair_y)
    internal_scores = pairwise_scores(model, train_x[holdout_indices])
    validation_scores = pairwise_scores(model, validation_x)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "pairs": len(pair_y) // 2,
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump(model, destination / f"{args.name}.joblib")
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
