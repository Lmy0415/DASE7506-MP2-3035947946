from pathlib import Path

import numpy as np

from common import parse_answer
from features import make_features
from ranker import Ranker


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("A model bundle is required")
    features = make_features(bank, arrays["embeddings"], "base")
    paths = sorted(Path(checkpoint).glob("member_*.npz"))
    if not paths:
        raise ValueError("No ensemble members found")
    scores = np.mean([Ranker.load(path).scores(features) for path in paths], axis=0)
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
