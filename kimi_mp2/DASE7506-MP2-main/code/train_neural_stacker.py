import argparse
import json
import time

import numpy as np

from baselines import correctness, describe
from common import DATA, read_jsonl
from ranker import train_ranker
from train_ranker import ROOT, cached_features, select, split_indices
from train_stacker import stack_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    with np.load(ROOT / "runs" / "cv_math_h256_r2_e20_scores.npz", allow_pickle=False) as saved:
        oof = saved["oof"]
        validation_neural = saved["validation"].mean(axis=0)
    train_x = stack_features(train_bank, features, oof)
    validation_x = stack_features(validation_bank, validation_features, validation_neural)
    train_indices, holdout_indices, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    history = []

    def evaluate(epoch, model):
        if epoch % 5 and epoch != args.epochs:
            return
        internal = describe(holdout_bank, holdout_references, select(holdout_bank, model.scores(train_x[holdout_indices])))
        validation = describe(validation_bank, validation_references, select(validation_bank, model.scores(validation_x)))
        history.append({"epoch": epoch, "internal": internal, "validation": validation})
        print(json.dumps({"epoch": epoch, "internal": internal["hard_accuracy"], "validation": validation["hard_accuracy"]}), flush=True)

    model = train_ranker(train_x, labels, train_indices, hidden_dim=args.hidden, seed=args.seed,
                         epochs=args.epochs, learning_rate=0.001, weight_decay=0.0001,
                         dropout=args.dropout, objective="ce", cosine=args.cosine, callback=evaluate)
    destination = ROOT / "runs" / "experiments"
    model.save(destination / f"{args.name}.npz")
    result = {"config": vars(args), "history": history, "seconds": time.perf_counter() - started}
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
