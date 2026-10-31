import json

import numpy as np

from baselines import correctness, describe
from common import DATA, read_jsonl
from ranker import train_ranker
from train_ranker import ROOT, cached_features, select, split_indices


def hard_weights(labels, indices, scores, factor):
    weights = np.ones_like(scores, dtype=np.float64)
    for i in indices:
        wrong = np.flatnonzero(~labels[i])
        if len(wrong):
            weights[i, wrong[np.argmax(scores[i, wrong])]] = factor
    return weights


def main():
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    destination = ROOT / "runs" / "experiments"
    result = {}

    def run(name, weights=None):
        model = train_ranker(features, labels, train_indices, hidden_dim=256, seed=17, epochs=20,
                             learning_rate=0.001, weight_decay=0.0001, dropout=0.1,
                             objective="ce", candidate_weights=weights)
        internal = model.scores(features[holdout_indices])
        validation = model.scores(validation_features)
        train_scores = model.scores(features)
        model.save(destination / f"{name}.npz")
        np.savez(destination / f"{name}_scores.npz", internal=internal, validation=validation)
        result[name] = {
            "internal": describe(holdout_bank, holdout_references, select(holdout_bank, internal)),
            "validation": describe(validation_bank, validation_references, select(validation_bank, validation)),
        }
        print(json.dumps({"name": name, "internal": result[name]["internal"]["hard_accuracy"], "validation": result[name]["validation"]["hard_accuracy"]}), flush=True)
        return train_scores

    baseline = run("neural_hard_baseline")
    hard2 = run("neural_hard_2", hard_weights(labels, train_indices, baseline, 2.0))
    run("neural_hard_4", hard_weights(labels, train_indices, baseline, 4.0))
    run("neural_hard_round3", hard_weights(labels, train_indices, hard2, 2.0))
    (destination / "neural_hard_experiments.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
