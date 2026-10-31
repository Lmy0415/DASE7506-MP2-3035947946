import argparse
import json

import numpy as np

from baselines import correctness, describe
from common import DATA, read_jsonl
from multitask_ranker import build_step_data, train_multitask
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    step_features, step_labels, step_mask = build_step_data(train_bank, features, labels)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    destination = ROOT / "runs" / "experiments"
    result = {}
    for alpha in args.alphas:
        name = f"multitask_{alpha}"
        model = train_multitask(features, labels, step_features, step_labels, step_mask,
                                train_indices, hidden_dim=256, seed=17, epochs=args.epochs,
                                learning_rate=0.001, weight_decay=0.0001, dropout=0.1, alpha=alpha)
        internal = model.scores(features[holdout_indices])
        validation = model.scores(validation_features)
        model.save(destination / f"{name}.npz")
        np.savez(destination / f"{name}_scores.npz", internal=internal, validation=validation)
        result[name] = {
            "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal)),
            "validation": describe(validation_bank, validation_references, select(validation_bank, validation)),
        }
        print(json.dumps({"alpha": alpha, "internal": result[name]["internal"]["hard_accuracy"], "validation": result[name]["validation"]["hard_accuracy"]}), flush=True)
    (destination / "multitask_experiments.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
