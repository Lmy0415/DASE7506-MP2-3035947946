import argparse
import json
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--hard-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    if args.hard_only:
        hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(train_bank)])
        train_indices = train_indices[hard[train_indices]]
    rows = (4 * train_indices[:, None] + np.arange(4)).ravel()
    embeddings = features[:, :, :1024].reshape(-1, 1024)
    pca = PCA(n_components=args.dimensions, svd_solver="randomized", random_state=1729)
    pca.fit(embeddings[rows])
    projected = pca.transform(embeddings)
    validation_projected = pca.transform(validation_features[:, :, :1024].reshape(-1, 1024))
    hand = features[:, :, 1024:].reshape(-1, features.shape[-1] - 1024)
    validation_hand = validation_features[:, :, 1024:].reshape(-1, features.shape[-1] - 1024)
    scaler = StandardScaler().fit(np.concatenate([projected[rows], hand[rows]], axis=1))
    x = scaler.transform(np.concatenate([projected, hand], axis=1))
    validation_x = scaler.transform(np.concatenate([validation_projected, validation_hand], axis=1))
    model = SVC(C=args.c, kernel="rbf", gamma="scale", cache_size=2000, class_weight="balanced")
    model.fit(x[rows], labels.reshape(-1)[rows])
    internal_scores = model.decision_function(x).reshape(-1, 4)[holdout_indices]
    validation_scores = model.decision_function(validation_x).reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "support_vectors": len(model.support_),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump({"pca": pca, "scaler": scaler, "model": model}, destination / f"{args.name}.joblib")
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "support": result["support_vectors"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
