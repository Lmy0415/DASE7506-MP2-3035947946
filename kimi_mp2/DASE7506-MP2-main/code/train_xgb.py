import argparse
import json
import time

import numpy as np
from xgboost import XGBRanker

from baselines import correctness, describe
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hand-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    if args.hand_only:
        features = features[:, :, 1024:]
        validation_features = validation_features[:, :, 1024:]
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    x = features[train_indices].reshape(-1, features.shape[-1])
    y = labels[train_indices].reshape(-1).astype(int)
    model = XGBRanker(objective="rank:pairwise", n_estimators=args.trees, max_depth=args.depth,
                      learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.7,
                      reg_lambda=10.0, min_child_weight=10, tree_method="hist", n_jobs=4,
                      random_state=args.seed)
    model.fit(x, y, group=np.full(len(train_indices), 4), verbose=False)
    internal_scores = model.predict(features[holdout_indices].reshape(-1, features.shape[-1])).reshape(-1, 4)
    validation_scores = model.predict(validation_features.reshape(-1, features.shape[-1])).reshape(-1, 4)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    result = {
        "config": vars(args),
        "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        "seconds": time.perf_counter() - started,
    }
    destination = ROOT / "runs" / "experiments"
    model.save_model(destination / f"{args.name}.json_model")
    np.savez(destination / f"{args.name}_scores.npz", internal=internal_scores, validation=validation_scores)
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"internal_hard": result["internal"]["hard_accuracy"], "validation_hard": result["validation"]["hard_accuracy"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
