import argparse
import json
import time
from pathlib import Path

import numpy as np

from baselines import answer_groups, describe, correctness
from common import DATA, load_inputs, read_jsonl
from features import make_features
from ranker import train_ranker

ROOT = Path(__file__).resolve().parents[1]


def split_indices(bank, seed=1729, fraction=0.2):
    hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(bank)])
    rng = np.random.default_rng(seed)
    holdout = []
    for mask in (hard, ~hard):
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        holdout.extend(indices[:round(len(indices) * fraction)])
    holdout = np.array(sorted(holdout))
    train = np.setdiff1d(np.arange(len(bank)), holdout)
    return train, holdout, hard


def select(bank, scores, consistency_rule=True):
    choices = scores.argmax(axis=1).astype(int)
    if consistency_rule:
        for i, answers in enumerate(answer_groups(bank)):
            valid_indices = [j for j, answer in enumerate(answers) if answer is not None]
            valid = [answers[j] for j in valid_indices]
            if valid and len(set(valid)) == 1:
                choices[i] = valid_indices[0]
            elif valid_indices and choices[i] not in valid_indices:
                choices[i] = max(valid_indices, key=lambda j: scores[i, j])
    return choices.tolist()


def cached_features(split, mode="base"):
    cache = ROOT / "runs" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split}_{mode}_v3_features.npz"
    bank = read_jsonl(DATA / "candidates" / f"gsm8k_{split}.jsonl")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            features = saved["features"].copy()
    else:
        _, arrays = load_inputs(split)
        features = make_features(bank, arrays["embeddings"], mode)
        np.savez(path, features=features)
    return bank, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--feature-mode", choices=("base", "centered", "nomath"), default="base")
    parser.add_argument("--objective", choices=("ce", "bce", "combined", "pairwise"), default="ce")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", args.feature_mode)
    validation_bank, validation_features = cached_features("validation", args.feature_mode)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, hard = split_indices(train_bank, args.split_seed)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    history = []

    def evaluate(epoch, model):
        if epoch % 5 and epoch != args.epochs:
            return
        train_scores = model.scores(features[holdout_indices])
        validation_scores = model.scores(validation_features)
        record = {
            "epoch": epoch,
            "internal": describe(holdout_bank, holdout_references, select(holdout_bank, train_scores)),
            "validation": describe(validation_bank, validation_references, select(validation_bank, validation_scores)),
        }
        history.append(record)
        print(json.dumps({"epoch": epoch, "internal_hard": record["internal"]["hard_accuracy"], "validation_hard": record["validation"]["hard_accuracy"]}), flush=True)

    model = train_ranker(
        features, labels, train_indices, hidden_dim=args.hidden, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, dropout=args.dropout,
        hard_only=args.hard_only, hard_mask=hard, objective=args.objective, cosine=args.cosine, callback=evaluate,
    )
    destination = ROOT / "runs" / "experiments"
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / f"{args.name}.npz")
    result = {
        "config": vars(args),
        "train_n": len(train_indices),
        "holdout_n": len(holdout_indices),
        "history": history,
        "seconds": time.perf_counter() - started,
        "asset_bytes": (destination / f"{args.name}.npz").stat().st_size,
    }
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"name": args.name, "seconds": result["seconds"], "asset_bytes": result["asset_bytes"]}))


if __name__ == "__main__":
    main()
