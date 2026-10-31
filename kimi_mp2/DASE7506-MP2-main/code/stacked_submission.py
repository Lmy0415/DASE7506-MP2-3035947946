from pathlib import Path

import joblib
import numpy as np

from common import parse_answer
from features import make_features
from ranker import Ranker
from train_stacker import stack_features


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("A stacked model bundle is required")
    checkpoint = Path(checkpoint)
    features = make_features(bank, arrays["embeddings"], "base")
    neural_paths = sorted((checkpoint / "neural").glob("member_*.npz"))
    if not neural_paths:
        raise ValueError("No neural members found")
    neural_scores = np.mean([Ranker.load(path).scores(features) for path in neural_paths], axis=0)
    stacker_paths = sorted(checkpoint.glob("stacker_*.joblib"))
    if not stacker_paths:
        raise ValueError("No stackers found")
    stackers = [joblib.load(path) for path in stacker_paths]
    levels = {61: "base", 211: "semantic", 225: "reach"}
    dimensions = {model.n_features_in_ for model in stackers}
    if len(dimensions) != 1:
        raise ValueError("Incompatible stacker dimensions")
    dimension = dimensions.pop()
    if dimension not in levels:
        raise ValueError("Unknown stacker dimensions")
    level = levels[dimension]
    stacked = stack_features(bank, features, neural_scores, level)
    flat = stacked.reshape(-1, stacked.shape[-1])
    scores = np.mean([model.predict_proba(flat)[:, 1] for model in stackers], axis=0).reshape(-1, 4)
    auxiliary_path = checkpoint / "auxiliary.npz"
    auxiliary_weight_path = checkpoint / "auxiliary_weight.txt"
    if auxiliary_path.exists():
        auxiliary = Ranker.load(auxiliary_path).scores(features)
        weight = float(auxiliary_weight_path.read_text()) if auxiliary_weight_path.exists() else 0.1
        scores = (1 - weight) * (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-6) + weight * (auxiliary - auxiliary.mean(axis=1, keepdims=True)) / np.maximum(auxiliary.std(axis=1, keepdims=True), 1e-6)
    calibrator_path = checkpoint / "calibrator.joblib"
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        scores = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1].reshape(scores.shape)
    smoothing_path = checkpoint / "answer_smoothing.txt"
    aggregation_path = checkpoint / "aggregation.txt"
    strength = float(smoothing_path.read_text()) if smoothing_path.exists() else 0.0
    if aggregation_path.exists() and aggregation_path.read_text().strip() == "mean":
        strength = 1.0
    if strength:
        for i, row in enumerate(bank):
            answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
            for answer in {answer for answer in answers if answer is not None}:
                members = [j for j, other in enumerate(answers) if other == answer]
                mean = scores[i, members].mean()
                scores[i, members] = (1 - strength) * scores[i, members] + strength * mean
    choices = scores.argmax(axis=1).astype(int).tolist()
    for i, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        valid_indices = [j for j, answer in enumerate(answers) if answer is not None]
        valid = [answers[j] for j in valid_indices]
        if valid and len(set(valid)) == 1:
            choices[i] = valid_indices[0]
        elif valid_indices and choices[i] not in valid_indices:
            choices[i] = max(valid_indices, key=lambda j: scores[i, j])
    return choices
