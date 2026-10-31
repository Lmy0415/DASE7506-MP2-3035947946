import argparse
import json
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def knn_scores(train_vectors, train_labels, query_vectors, k, temperature, batch_size=256):
    output = []
    for start in range(0, len(query_vectors), batch_size):
        similarities = query_vectors[start:start + batch_size] @ train_vectors.T
        indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        selected_similarities = np.take_along_axis(similarities, indices, axis=1)
        weights = np.exp((selected_similarities - selected_similarities.max(axis=1, keepdims=True)) / temperature)
        labels = train_labels[indices]
        output.append((weights * labels).sum(axis=1) / weights.sum(axis=1))
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--all-groups", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    if not args.all_groups:
        hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(train_bank)])
        train_indices = train_indices[hard[train_indices]]
    embeddings = features[:, :, :1024].reshape(-1, 1024)
    validation_embeddings = validation_features[:, :, :1024].reshape(-1, 1024)
    fit_rows = (4 * train_indices[:, None] + np.arange(4)).ravel()
    pca = PCA(n_components=args.dimensions, svd_solver="randomized", random_state=1729)
    pca.fit(embeddings[fit_rows])
    transformed = normalize(pca.transform(embeddings).astype(np.float32))
    validation_transformed = normalize(pca.transform(validation_embeddings).astype(np.float32))
    train_vectors = transformed[fit_rows]
    train_labels = labels.reshape(-1)[fit_rows].astype(np.float32)
    query_rows = (4 * holdout_indices[:, None] + np.arange(4)).ravel()
    internal_scores = knn_scores(train_vectors, train_labels, transformed[query_rows], args.k, args.temperature).reshape(-1, 4)
    validation_scores = knn_scores(train_vectors, train_labels, validation_transformed, args.k, args.temperature).reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump({"pca": pca, "vectors": train_vectors, "labels": train_labels}, destination / f"{args.name}.joblib")
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
