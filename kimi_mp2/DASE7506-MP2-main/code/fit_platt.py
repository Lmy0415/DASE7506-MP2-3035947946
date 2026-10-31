import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from baselines import correctness
from common import DATA, read_jsonl
from train_ranker import ROOT, cached_features, split_indices


def main():
    train_bank, _ = cached_features("train", "base")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    _, holdout, _ = split_indices(train_bank)
    holdout_bank = [train_bank[i] for i in holdout]
    labels = correctness(holdout_bank, [references[i] for i in holdout])
    saved = [np.load(ROOT / "runs" / "experiments" / f"{name}_scores.npz") for name in ("stack_hgb_all", "stack_hgb_leaf10")]
    scores = np.mean([item["internal"] for item in saved], axis=0)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels.reshape(-1))
    destination = ROOT / "runs" / "candidate_calibrated"
    (destination / "neural").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / "calibrator.joblib")
    (destination / "aggregation.txt").write_text("mean\n")
    print({"coefficient": model.coef_.tolist(), "intercept": model.intercept_.tolist()})


if __name__ == "__main__":
    main()
