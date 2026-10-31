import argparse
import json

import numpy as np

from baselines import describe
from common import DATA, read_jsonl
from evaluate_ensemble import smooth_by_answer
from ranker import Ranker
from train_ranker import ROOT, cached_features, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    bank, features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    paths = sorted((ROOT / "runs" / args.bundle).glob("member_*.npz"))[:args.count]
    scores = np.mean([Ranker.load(path).scores(features) for path in paths], axis=0)
    result = {}
    for strength in (0, 0.25, 0.5, 0.75, 1.0):
        adjusted = smooth_by_answer(bank, scores, strength)
        name = str(strength)
        result[name] = describe(bank, references, select(bank, adjusted))
        print(json.dumps({"strength": strength, "hard": result[name]["hard_accuracy"], "all": result[name]["accuracy"]}))
    if args.output:
        (ROOT / "runs" / args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
