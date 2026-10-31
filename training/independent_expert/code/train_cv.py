import argparse
import json
import time

import numpy as np

from baselines import answer_groups, correctness, describe
from common import DATA, read_jsonl
from ranker import train_ranker
from train_ranker import ROOT, cached_features, select

SEEDS = (17, 29, 41, 59, 73, 83, 97, 109, 127, 139)


def fold_assignments(bank, folds, repeat):
    hard = np.array([len({answer for answer in row if answer is not None}) > 1 for row in answer_groups(bank)])
    rng = np.random.default_rng(1729 + repeat * 1009)
    assignment = np.empty(len(bank), dtype=int)
    for mask in (hard, ~hard):
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, folds)):
            assignment[part] = fold
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cv_h256_r1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--feature-mode", choices=("base", "nomath"), default="base")
    parser.add_argument("--cosine", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    train_bank, features = cached_features("train", args.feature_mode)
    validation_bank, validation_features = cached_features("validation", args.feature_mode)
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    destination = ROOT / "runs" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    validation_scores = []
    oof_sum = np.zeros((len(train_bank), 4), dtype=np.float64)
    oof_count = np.zeros(len(train_bank), dtype=int)
    records = []
    member = 0
    for repeat in range(args.repeats):
        assignment = fold_assignments(train_bank, args.folds, repeat)
        for fold in range(args.folds):
            holdout = np.flatnonzero(assignment == fold)
            train_indices = np.flatnonzero(assignment != fold)
            seed = SEEDS[member]
            model = train_ranker(features, labels, train_indices, hidden_dim=args.hidden,
                                 seed=seed, epochs=args.epochs, learning_rate=0.001,
                                 weight_decay=0.0001, dropout=0.1, objective="ce", cosine=args.cosine)
            filename = f"member_r{repeat}_f{fold}.npz"
            model.save(destination / filename)
            oof_sum[holdout] += model.scores(features[holdout])
            oof_count[holdout] += 1
            score = model.scores(validation_features)
            validation_scores.append(score)
            record = {"repeat": repeat, "fold": fold, "seed": seed,
                      "validation": describe(validation_bank, validation_references, select(validation_bank, score))}
            records.append(record)
            print(json.dumps({"member": member, "validation_hard": record["validation"]["hard_accuracy"]}), flush=True)
            member += 1
    oof_scores = oof_sum / np.maximum(oof_count, 1)[:, None]
    result = {
        "config": vars(args),
        "members": records,
        "oof_covered": int((oof_count > 0).sum()),
        "oof": describe(train_bank, references, select(train_bank, oof_scores)),
        "validation": describe(validation_bank, validation_references, select(validation_bank, np.mean(validation_scores, axis=0))),
        "training_seconds": time.perf_counter() - started,
        "asset_bytes": sum(path.stat().st_size for path in destination.glob("member_*.npz")),
    }
    np.savez(ROOT / "runs" / f"{args.name}_scores.npz", oof=oof_scores,
             validation=np.asarray(validation_scores), oof_count=oof_count)
    (ROOT / "runs" / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ensemble_validation_hard": result["validation"]["hard_accuracy"], "ensemble_validation_all": result["validation"]["accuracy"], "oof_hard": result["oof"]["hard_accuracy"], "asset_bytes": result["asset_bytes"]}))


if __name__ == "__main__":
    main()
