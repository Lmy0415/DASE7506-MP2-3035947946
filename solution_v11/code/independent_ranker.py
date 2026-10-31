from pathlib import Path

import numpy as np


class Ranker:
    def __init__(self, input_dim, hidden_dim=128, seed=17):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": (rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2 / input_dim)).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": (rng.standard_normal((hidden_dim, 1)) * np.sqrt(1 / hidden_dim)).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.scale = np.ones(input_dim, dtype=np.float32)

    def fit_normalizer(self, features):
        flat = features.reshape(-1, features.shape[-1])
        self.mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-5] = 1.0

    def normalize(self, features):
        return (features - self.mean) / self.scale

    def scores(self, features, batch_size=512):
        output = []
        for start in range(0, len(features), batch_size):
            batch = self.normalize(features[start:start + batch_size])
            flat = batch.reshape(-1, batch.shape[-1])
            hidden = np.maximum(flat @ self.params["W1"] + self.params["b1"], 0)
            scores = (hidden @ self.params["W2"] + self.params["b2"]).reshape(len(batch), 4)
            output.append(scores)
        return np.concatenate(output)

    def save(self, path):
        np.savez(Path(path), mean=self.mean, scale=self.scale, **self.params)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as asset:
            instance = cls(asset["W1"].shape[0], asset["W1"].shape[1])
            instance.mean = asset["mean"].copy()
            instance.scale = asset["scale"].copy()
            instance.params = {name: asset[name].copy() for name in ("W1", "b1", "W2", "b2")}
        return instance


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def train_ranker(features, labels, train_indices, hidden_dim=128, seed=17, epochs=60,
                 batch_size=256, learning_rate=0.001, weight_decay=0.0001,
                 dropout=0.1, hard_only=False, hard_mask=None, objective="ce", cosine=False, callback=None):
    model = Ranker(features.shape[-1], hidden_dim, seed)
    model.fit_normalizer(features[train_indices])
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.params.items()}
    step = 0
    if hard_only:
        train_indices = np.asarray(train_indices)[hard_mask[train_indices]]
    total_steps = epochs * int(np.ceil(len(train_indices) / batch_size))
    for epoch in range(epochs):
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = model.normalize(features[indices])
            n = len(indices)
            flat = batch.reshape(-1, batch.shape[-1])
            pre_activation = flat @ model.params["W1"] + model.params["b1"]
            hidden = np.maximum(pre_activation, 0)
            if dropout:
                mask = (rng.random(hidden.shape) >= dropout).astype(np.float32) / (1 - dropout)
                hidden_dropped = hidden * mask
            else:
                mask = None
                hidden_dropped = hidden
            scores = (hidden_dropped @ model.params["W2"] + model.params["b2"]).reshape(n, 4)
            binary_targets = labels[indices].astype(np.float32)
            if objective in ("ce", "combined"):
                targets = binary_targets / binary_targets.sum(axis=1, keepdims=True)
                score_gradient = (softmax(scores) - targets) / n
            else:
                score_gradient = np.zeros_like(scores)
            if objective in ("bce", "combined"):
                probabilities = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
                weight = 0.5 if objective == "combined" else 1.0
                score_gradient += weight * (probabilities - binary_targets) / (4 * n)
            if objective == "pairwise":
                negative = 1 - binary_targets
                differences = scores[:, None, :] - scores[:, :, None]
                pair_weights = 1 / (1 + np.exp(-np.clip(differences, -30, 30)))
                pair_weights *= binary_targets[:, :, None] * negative[:, None, :]
                pair_counts = np.maximum(binary_targets.sum(axis=1) * negative.sum(axis=1), 1)
                score_gradient = (pair_weights.sum(axis=1) * negative - pair_weights.sum(axis=2) * binary_targets) / pair_counts[:, None] / n
            gradients = {}
            gradients["W2"] = hidden_dropped.T @ score_gradient.reshape(-1, 1) + weight_decay * model.params["W2"]
            gradients["b2"] = score_gradient.sum().reshape(1)
            hidden_gradient = score_gradient.reshape(-1, 1) @ model.params["W2"].T
            if mask is not None:
                hidden_gradient *= mask
            hidden_gradient[pre_activation <= 0] = 0
            gradients["W1"] = flat.T @ hidden_gradient + weight_decay * model.params["W1"]
            gradients["b1"] = hidden_gradient.sum(axis=0)
            step += 1
            for name, parameter in model.params.items():
                gradient = np.clip(gradients[name], -5, 5)
                first, second = moments[name]
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient * gradient
                first_hat = first / (1 - 0.9 ** step)
                second_hat = second / (1 - 0.999 ** step)
                rate = learning_rate * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / total_steps))) if cosine else learning_rate
                parameter -= rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        if callback is not None:
            callback(epoch + 1, model)
    return model
