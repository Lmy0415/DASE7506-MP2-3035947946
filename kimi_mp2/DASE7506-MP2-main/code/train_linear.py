import argparse
import json
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices


def pairs(features, labels, indices):
    rows = []
    for i in indices:
        for c in np.flatnonzero(labels[i]):
            for w in np.flatnonzero(~labels[i]):
                rows.append(features[i, c] - features[i, w])
    positive = np.asarray(rows, dtype=np.float32)
    return np.concatenate([positive, -positive]), np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--c", type=float, default=0.001)
    parser.add_argument("--hand-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    if args.hand_only:
        features = features[:, :, -33:]
        validation_features = validation_features[:, :, -33:]
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    scaler = StandardScaler().fit(features[train_indices].reshape(-1, features.shape[-1]))
    normalized = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    validation_normalized = scaler.transform(validation_features.reshape(-1, features.shape[-1])).reshape(validation_features.shape)
    pair_train, pair_labels = pairs(normalized, labels, train_indices)
    model = LinearSVC(C=args.c, max_iter=10000)
    model.fit(pair_train, pair_labels)
    internal_scores = model.decision_function(normalized.reshape(-1, features.shape[-1])).reshape(-1, 4)[holdout_indices]
    validation_scores = model.decision_function(validation_normalized.reshape(-1, features.shape[-1])).reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump({"scaler": scaler, "model": model}, destination / f"{args.name}.joblib")
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
