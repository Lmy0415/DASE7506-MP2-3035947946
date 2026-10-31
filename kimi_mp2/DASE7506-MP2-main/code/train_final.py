import argparse
import json
import time

import numpy as np

from baselines import describe, correctness
from common import DATA, read_jsonl
from ranker import train_ranker
from train_ranker import ROOT, cached_features, select

SEEDS = (17, 29, 41, 59, 73)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--objective", choices=("ce", "bce", "combined"), default="ce")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--name", default="model_bundle")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    destination = ROOT / "runs" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    scores = []
    for seed in SEEDS:
        model = train_ranker(
            features, labels, np.arange(len(train_bank)), hidden_dim=args.hidden,
            seed=seed, epochs=args.epochs, learning_rate=0.001,
            weight_decay=0.0001, dropout=0.1, objective=args.objective, cosine=args.cosine,
        )
        model.save(destination / f"member_{seed}.npz")
        scores.append(model.scores(validation_features))
        print(json.dumps({"seed": seed, "validation_hard": describe(validation_bank, validation_references, select(validation_bank, scores[-1]))["hard_accuracy"]}), flush=True)
    result = {
        "seeds": SEEDS,
        "epochs": args.epochs,
        "hidden_dim": args.hidden,
        "feature_mode": "base",
        "training_questions": len(train_bank),
        "training_seconds": time.perf_counter() - started,
        "validation": {},
    }
    for count in (3, len(SEEDS)):
        name = f"first_{count}_mean"
        choices = select(validation_bank, np.mean(scores[:count], axis=0))
        result["validation"][name] = describe(validation_bank, validation_references, choices)
        print(json.dumps({"ensemble": name, "validation_hard": result["validation"][name]["hard_accuracy"], "validation_all": result["validation"][name]["accuracy"]}))
    result["asset_bytes"] = sum(path.stat().st_size for path in destination.glob("member_*.npz"))
    (ROOT / "runs" / f"{args.name}_training.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
