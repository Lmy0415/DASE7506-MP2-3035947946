import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from ranker import softmax
from train_ranker import ROOT, cached_features, select, split_indices


def stack_features(bank, features, scores, level="reach"):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    zscore = (scores - mean) / np.maximum(std, 1e-6)
    probabilities = softmax(scores)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1) / 3.0
    cluster = scores.copy()
    for i, answers in enumerate(answer_groups(bank)):
        for answer in {answer for answer in answers if answer is not None}:
            members = [j for j, other in enumerate(answers) if other == answer]
            cluster[i, members] = scores[i, members].mean()
    cluster_z = (cluster - cluster.mean(axis=1, keepdims=True)) / np.maximum(cluster.std(axis=1, keepdims=True), 1e-6)
    rank_features = np.stack([
        scores, zscore, probabilities, ranks, cluster, cluster_z,
        scores - scores.max(axis=1, keepdims=True),
        zscore * features[:, :, 1024 + 1],
        zscore * features[:, :, -6],
    ], axis=2)
    group = np.concatenate([mean, std, scores.max(axis=1, keepdims=True), scores.min(axis=1, keepdims=True)], axis=1)
    group = np.repeat(group[:, None, :], 4, axis=1)
    parts = [rank_features, features[:, :, 1024:]]
    if level in ("semantic", "reach"):
        from semantic_features import semantic_features
        parts.append(semantic_features(bank))
    if level == "reach":
        from reachability import reachability_features
        parts.append(reachability_features(bank))
    if level not in ("base", "semantic", "reach"):
        raise ValueError(f"Unknown stack feature level: {level}")
    parts.append(group)
    return np.concatenate(parts, axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", choices=("hgb", "extra"), default="hgb")
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--leaf", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--base-scores", default="cv_math_h256_r2_e20_scores.npz")
    parser.add_argument("--level", choices=("base", "semantic", "reach"), default="reach")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    with np.load(ROOT / "runs" / args.base_scores, allow_pickle=False) as saved:
        oof = saved["oof"]
        validation_scores = saved["validation"].mean(axis=0)
    train_x = stack_features(train_bank, features, oof, args.level)
    validation_x = stack_features(validation_bank, validation_features, validation_scores, args.level)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(train_bank)])
    if args.hard_only:
        train_indices = train_indices[hard[train_indices]]
    rows = (4 * train_indices[:, None] + np.arange(4)).ravel()
    x = train_x.reshape(-1, train_x.shape[-1])[rows]
    y = labels.reshape(-1)[rows]
    weights = np.repeat(np.where(hard[train_indices], args.hard_weight, 1.0), 4)
    if args.model == "hgb":
        model = HistGradientBoostingClassifier(max_iter=args.iterations, learning_rate=args.learning_rate,
                                               max_leaf_nodes=args.nodes, min_samples_leaf=args.leaf,
                                               l2_regularization=args.l2, random_state=1729)
    else:
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=args.leaf, max_features=0.7,
                                     n_jobs=4, class_weight="balanced", random_state=1729)
    model.fit(x, y, sample_weight=weights)
    internal_scores = model.predict_proba(train_x.reshape(-1, train_x.shape[-1]))[:, 1].reshape(-1, 4)[holdout_indices]
    validation_scores = model.predict_proba(validation_x.reshape(-1, validation_x.shape[-1]))[:, 1].reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
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
