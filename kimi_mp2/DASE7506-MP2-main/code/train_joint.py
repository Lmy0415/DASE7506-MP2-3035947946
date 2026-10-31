import argparse
import json
import time

import numpy as np
import torch

from baselines import correctness, describe
from common import DATA, read_jsonl
from joint_ranker import JointRanker, joint_scores, save_joint
from train_ranker import ROOT, cached_features, select, split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = correctness(train_bank, references)
    train_indices, holdout_indices, _ = split_indices(train_bank, args.split_seed)
    holdout_bank = [train_bank[i] for i in holdout_indices]
    holdout_references = [references[i] for i in holdout_indices]
    model = JointRanker(features.shape[-1], args.hidden, args.heads, args.dropout)
    flat = features[train_indices].reshape(-1, features.shape[-1])
    model.feature_mean.copy_(torch.from_numpy(flat.mean(axis=0, dtype=np.float64).astype(np.float32)))
    scale = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    model.feature_scale.copy_(torch.from_numpy(scale))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    tensor = torch.from_numpy(features)
    targets_all = labels.astype(np.float32)
    targets_all /= targets_all.sum(axis=1, keepdims=True)
    targets = torch.from_numpy(targets_all)
    rng = np.random.default_rng(args.seed + 1)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        for start in range(0, len(order), 256):
            indices = torch.from_numpy(order[start:start + 256])
            optimizer.zero_grad()
            scores = model(tensor[indices])
            loss = -(targets[indices] * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            record = {
                "epoch": epoch,
                "internal": describe(holdout_bank, holdout_references, select(holdout_bank, joint_scores(model, features[holdout_indices]))),
                "validation": describe(validation_bank, validation_references, select(validation_bank, joint_scores(model, validation_features))),
            }
            history.append(record)
            print(json.dumps({"epoch": epoch, "internal_hard": record["internal"]["hard_accuracy"], "validation_hard": record["validation"]["hard_accuracy"]}), flush=True)
    destination = ROOT / "runs" / "experiments"
    save_joint(model, destination / f"{args.name}.npz")
    result = {"config": vars(args), "history": history, "seconds": time.perf_counter() - started,
              "asset_bytes": (destination / f"{args.name}.npz").stat().st_size}
    (destination / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
