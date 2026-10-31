import argparse
import json

import numpy as np

from baselines import describe
from common import DATA, read_jsonl
from ranker import Ranker
from train_ranker import ROOT, cached_features, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    bank, features = cached_features("validation", "base")
    references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    bundle_scores = []
    result = {"bundles": {}, "combined": None}
    for name in args.names:
        paths = sorted((ROOT / "runs" / name).glob("member_*.npz"))
        scores = [Ranker.load(path).scores(features) for path in paths]
        mean = np.mean(scores, axis=0)
        bundle_scores.append(mean)
        result["bundles"][name] = describe(bank, references, select(bank, mean))
        print(json.dumps({"bundle": name, "hard": result["bundles"][name]["hard_accuracy"], "all": result["bundles"][name]["accuracy"]}))
    combined = np.mean(bundle_scores, axis=0)
    result["combined"] = describe(bank, references, select(bank, combined))
    print(json.dumps({"bundle": "combined", "hard": result["combined"]["hard_accuracy"], "all": result["combined"]["accuracy"]}))
    if args.output:
        (ROOT / "runs" / args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
