from pathlib import Path

import numpy as np

from ranker import softmax


class DeepRanker:
    def __init__(self, input_dim, hidden_dims=(256, 128), seed=17):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.params = {
            "W1": (rng.standard_normal((input_dim, h1)) * np.sqrt(2 / input_dim)).astype(np.float32),
            "b1": np.zeros(h1, dtype=np.float32),
            "W2": (rng.standard_normal((h1, h2)) * np.sqrt(2 / h1)).astype(np.float32),
            "b2": np.zeros(h2, dtype=np.float32),
            "W3": (rng.standard_normal((h2, 1)) * np.sqrt(1 / h2)).astype(np.float32),
            "b3": np.zeros(1, dtype=np.float32),
        }
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.scale = np.ones(input_dim, dtype=np.float32)

    def fit_normalizer(self, features):
        flat = features.reshape(-1, features.shape[-1])
        self.mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-5] = 1.0

    def scores(self, features, batch_size=512):
        output = []
        for start in range(0, len(features), batch_size):
            batch = (features[start:start + batch_size] - self.mean) / self.scale
            flat = batch.reshape(-1, batch.shape[-1])
            h1 = np.maximum(flat @ self.params["W1"] + self.params["b1"], 0)
            h2 = np.maximum(h1 @ self.params["W2"] + self.params["b2"], 0)
            output.append((h2 @ self.params["W3"] + self.params["b3"]).reshape(len(batch), 4))
        return np.concatenate(output)

    def save(self, path):
        np.savez(Path(path), mean=self.mean, scale=self.scale, **self.params)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as asset:
            instance = cls(asset["W1"].shape[0], (asset["W1"].shape[1], asset["W2"].shape[1]))
            instance.mean = asset["mean"].copy()
            instance.scale = asset["scale"].copy()
            instance.params = {name: asset[name].copy() for name in ("W1", "b1", "W2", "b2", "W3", "b3")}
        return instance


def train_deep_ranker(features, labels, train_indices, hidden_dims=(256, 128), seed=17,
                      epochs=20, batch_size=256, learning_rate=0.001,
                      weight_decay=0.0001, dropout=0.1, callback=None):
    model = DeepRanker(features.shape[-1], hidden_dims, seed)
    model.fit_normalizer(features[train_indices])
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.params.items()}
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = (features[indices] - model.mean) / model.scale
            n = len(indices)
            flat = batch.reshape(-1, batch.shape[-1])
            z1 = flat @ model.params["W1"] + model.params["b1"]
            h1 = np.maximum(z1, 0)
            mask1 = (rng.random(h1.shape) >= dropout).astype(np.float32) / (1 - dropout)
            d1 = h1 * mask1
            z2 = d1 @ model.params["W2"] + model.params["b2"]
            h2 = np.maximum(z2, 0)
            mask2 = (rng.random(h2.shape) >= dropout).astype(np.float32) / (1 - dropout)
            d2 = h2 * mask2
            scores = (d2 @ model.params["W3"] + model.params["b3"]).reshape(n, 4)
            targets = labels[indices].astype(np.float32)
            targets /= targets.sum(axis=1, keepdims=True)
            ds = (softmax(scores) - targets) / n
            gradients = {}
            gradients["W3"] = d2.T @ ds.reshape(-1, 1) + weight_decay * model.params["W3"]
            gradients["b3"] = ds.sum().reshape(1)
            dh2 = ds.reshape(-1, 1) @ model.params["W3"].T
            dh2 *= mask2
            dh2[z2 <= 0] = 0
            gradients["W2"] = d1.T @ dh2 + weight_decay * model.params["W2"]
            gradients["b2"] = dh2.sum(axis=0)
            dh1 = dh2 @ model.params["W2"].T
            dh1 *= mask1
            dh1[z1 <= 0] = 0
            gradients["W1"] = flat.T @ dh1 + weight_decay * model.params["W1"]
            gradients["b1"] = dh1.sum(axis=0)
            step += 1
            for name, parameter in model.params.items():
                gradient = np.clip(gradients[name], -5, 5)
                first, second = moments[name]
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient * gradient
                parameter -= learning_rate * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
        if callback is not None:
            callback(epoch + 1, model)
    return model
