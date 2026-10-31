import argparse
import json
import re
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def documents(bank):
    return [NUMBER.sub(" num ", candidate["text"].lower()) for row in bank for candidate in row["candidates"]]


def pair_differences(matrix, labels, indices):
    differences = []
    for i in indices:
        correct = np.flatnonzero(labels[i])
        wrong = np.flatnonzero(~labels[i])
        for c in correct:
            for w in wrong:
                differences.append(matrix[4 * i + c] - matrix[4 * i + w])
    positive = sparse.vstack(differences, format="csr")
    negative = -positive
    return sparse.vstack([positive, negative], format="csr"), np.concatenate([np.ones(len(differences)), np.zeros(len(differences))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--hand-weight", type=float, default=0.1)
    parser.add_argument("--no-hand", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=args.max_features,
                                 sublinear_tf=True, strip_accents="unicode")
    train_text = vectorizer.fit_transform(documents(train_bank))
    validation_text = vectorizer.transform(documents(validation_bank))
    scaler = None
    if not args.no_hand:
        scaler = StandardScaler().fit(features[train_indices, :, -33:].reshape(-1, 33))
        train_hand = scaler.transform(features[:, :, -33:].reshape(-1, 33)) * args.hand_weight
        validation_hand = scaler.transform(validation_features[:, :, -33:].reshape(-1, 33)) * args.hand_weight
        train_text = sparse.hstack([train_text, sparse.csr_matrix(train_hand)], format="csr")
        validation_text = sparse.hstack([validation_text, sparse.csr_matrix(validation_hand)], format="csr")
    pair_train, pair_labels = pair_differences(train_text, labels, train_indices)
    model = LinearSVC(C=args.c, max_iter=10000)
    model.fit(pair_train, pair_labels)
    internal_scores = model.decision_function(train_text).reshape(-1, 4)[holdout_indices]
    validation_scores = model.decision_function(validation_text).reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "vocabulary": len(vectorizer.vocabulary_),
        "pairs": len(pair_labels) // 2,
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    joblib.dump({"vectorizer": vectorizer, "scaler": scaler, "model": model}, destination / f"{args.name}.joblib")
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "validation_all": result["validation"]["accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
