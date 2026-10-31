import argparse
import json
import time

import numpy as np

from baselines import correctness, describe
from common import DATA, read_jsonl
from deep_ranker import train_deep_ranker
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank, args.split_seed)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    history = []

    def evaluate(epoch, model):
        if epoch % 5 and epoch != args.epochs:
            return
        record = {
            "epoch": epoch,
            "internal": describe(holdout_bank, holdout_references, select(holdout_bank, model.scores(features[holdout_indices]))),
            "validation": describe(validation_bank, validation_references, select(validation_bank, model.scores(validation_features))),
        }
        history.append(record)
        print(json.dumps({"epoch": epoch, "internal_hard": record["internal"]["hard_accuracy"], "validation_hard": record["validation"]["hard_accuracy"]}), flush=True)

    model = train_deep_ranker(features, labels, train_indices, (args.hidden1, args.hidden2), args.seed,
                              args.epochs, learning_rate=args.learning_rate, dropout=args.dropout,
                              callback=evaluate)
    destination = ROOT / "runs" / "experiments"
    model.save(destination / f"{args.name}.npz")
    result = {"config": vars(args), "history": history, "seconds": time.perf_counter() - started,
              "asset_bytes": (destination / f"{args.name}.npz").stat().st_size}
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
