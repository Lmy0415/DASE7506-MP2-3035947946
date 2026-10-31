import json

import numpy as np

from baselines import describe
from common import DATA, read_jsonl
from ranker import Ranker
from train_ranker import ROOT, cached_features, select, split_indices

NAMES = ["math_h256_s17_e20", "math_h256_s29_e20", "math_h256_s41_e20", "math_h256_s59_e20", "math_h256_s73_e20"]


def main():
    train_bank, features = cached_features("train", "base")
    validation_bank, validation_features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    validation_references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    _, holdout, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout]
    holdout_references = [references[i] for i in holdout]
    internal_scores = np.mean([Ranker.load(ROOT / "runs" / "experiments" / f"{name}.npz").scores(features[holdout]) for name in NAMES], axis=0)
    paths = sorted((ROOT / "runs" / "full_math_h256_e20").glob("member_*.npz"))[:3]
    validation_scores = np.mean([Ranker.load(path).scores(validation_features) for path in paths], axis=0)
    internal_math = features[holdout, :, -15:]
    validation_math = validation_features[:, :, -15:]
    result = {"internal": {}, "validation": {}}
    for feature, offset in (("none", None), ("invalid_ratio", 2), ("any_invalid", 3), ("final_contradiction", 10), ("tail_invalid", 11)):
        for strength in (0.1, 0.25, 0.5, 1.0, 2.0):
            if offset is None and strength != 0.1:
                continue
            internal = internal_scores.copy()
            validation = validation_scores.copy()
            if offset is not None:
                internal -= strength * internal_math[:, :, offset]
                validation -= strength * validation_math[:, :, offset]
            name = f"{feature}_{strength}"
            result["internal"][name] = describe(holdout_bank, holdout_references, select(holdout_bank, internal))
            result["validation"][name] = describe(validation_bank, validation_references, select(validation_bank, validation))
            print(json.dumps({"rule": name, "internal": result["internal"][name]["hard_accuracy"], "validation": result["validation"][name]["hard_accuracy"]}))
    (ROOT / "runs" / "math_calibration.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
