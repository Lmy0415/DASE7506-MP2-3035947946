import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer, validate_choices
from submission import predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "model_bundle")
    args = parser.parse_args()
    bank, arrays = load_inputs("validation")
    checkpoint = args.checkpoint
    subset = copy.deepcopy(bank[:7])
    subset_arrays = {"embeddings": arrays["embeddings"][:7].copy()}
    first = predict(subset, subset_arrays, checkpoint)
    second = predict(subset, subset_arrays, checkpoint)
    validate_choices(subset, [row["id"] for row in subset], first)
    single = predict(copy.deepcopy(bank[:1]), {"embeddings": arrays["embeddings"][:1].copy()}, checkpoint)
    validate_choices(bank[:1], [bank[0]["id"]], single)
    permutation = [1, 2, 3, 0]
    permuted = copy.deepcopy(subset)
    for row in permuted:
        row["candidates"] = [row["candidates"][i] for i in permutation]
        for index, candidate in enumerate(row["candidates"]):
            candidate["index"] = index
    permuted_arrays = {"embeddings": subset_arrays["embeddings"][:, permutation].copy()}
    permuted_choices = predict(permuted, permuted_arrays, checkpoint)
    mapped = [permutation[choice] for choice in permuted_choices]
    answer_mismatches = sum(
        parse_answer(row["candidates"][original]["text"]) != parse_answer(row["candidates"][reordered]["text"])
        for row, original, reordered in zip(subset, first, mapped)
    )
    paths = sorted(checkpoint.glob("member_*.npz"))
    result = {
        "subset_n": len(subset),
        "single_n": len(single),
        "valid_python_indices": all(type(choice) is int and 0 <= choice < 4 for choice in first + single),
        "deterministic": first == second,
        "candidate_permutation_mismatches": sum(a != b for a, b in zip(first, mapped)),
        "selected_answer_permutation_mismatches": answer_mismatches,
        "asset_bytes": sum(path.stat().st_size for path in paths),
        "asset_limit_bytes": 16 * 1024 * 1024,
        "hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    result["passed"] = (
        result["valid_python_indices"] and result["deterministic"]
        and result["selected_answer_permutation_mismatches"] == 0
        and result["asset_bytes"] <= result["asset_limit_bytes"]
    )
    (ROOT / "runs" / "contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
