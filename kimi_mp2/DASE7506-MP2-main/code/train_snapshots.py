import argparse
import json

import numpy as np

from baselines import correctness, describe
from common import DATA, read_jsonl
from ranker import train_ranker
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    destination = ROOT / "runs" / f"snapshots_s{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    internal_scores = {}
    validation_scores = {}

    def callback(epoch, model):
        if epoch >= 20 and epoch % 5 == 0:
            model.save(destination / f"epoch_{epoch}.npz")
            internal_scores[epoch] = model.scores(features[holdout_indices])
            validation_scores[epoch] = model.scores(validation_features)

    train_ranker(features, labels, train_indices, hidden_dim=256, seed=args.seed,
                 epochs=args.epochs, learning_rate=0.001, weight_decay=0.0001,
                 dropout=0.1, objective="ce", cosine=True, callback=callback)
    result = {"internal": {}, "validation": {}}
    combinations = {
        "last3": [40, 45, 50],
        "last4": [35, 40, 45, 50],
        "last5_with20": [20, 35, 40, 45, 50],
        "all_late": [20, 25, 30, 35, 40, 45, 50],
        "last2": [45, 50],
    }
    for name, epochs in combinations.items():
        internal = np.mean([internal_scores[epoch] for epoch in epochs], axis=0)
        validation = np.mean([validation_scores[epoch] for epoch in epochs], axis=0)
        result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal))
        result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation))
        print(json.dumps({"seed": args.seed, "combination": name, "internal": result["internal"][name]["hard_accuracy"], "validation": result["validation"][name]["hard_accuracy"]}))
    np.savez(ROOT / "runs" / f"snapshots_s{args.seed}_scores.npz",
             **{f"internal_{epoch}": value for epoch, value in internal_scores.items()},
             **{f"validation_{epoch}": value for epoch, value in validation_scores.items()})
    (ROOT / "runs" / f"snapshots_s{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
