import numpy as np

from ranker import Ranker, softmax
from step_verifier import step_rows


def build_step_data(bank, features, labels, max_steps=12):
    rows, targets, groups = step_rows(bank, features, labels)
    n = len(bank)
    step_features = np.zeros((n, 4, max_steps, 16), dtype=np.float32)
    step_labels = np.zeros((n, 4, max_steps), dtype=np.float32)
    mask = np.zeros((n, 4, max_steps), dtype=np.float32)
    for question_index, group in enumerate(groups):
        for candidate_index, indices in enumerate(group):
            chosen = indices[:max_steps]
            if not chosen:
                continue
            step_features[question_index, candidate_index, :len(chosen)] = rows[chosen, -16:]
            step_labels[question_index, candidate_index, :len(chosen)] = targets[chosen]
            mask[question_index, candidate_index, :len(chosen)] = 1.0
    return step_features, step_labels, mask


def train_multitask(features, labels, step_features, step_labels, step_mask, train_indices,
                    hidden_dim=256, seed=17, epochs=20, batch_size=256,
                    learning_rate=0.001, weight_decay=0.0001, dropout=0.1,
                    alpha=0.1, callback=None):
    model = Ranker(features.shape[-1], hidden_dim, seed)
    rng = np.random.default_rng(seed + 1)
    model.params["W3"] = (rng.standard_normal((hidden_dim, 1)) * np.sqrt(1 / hidden_dim)).astype(np.float32)
    model.params["U3"] = (rng.standard_normal((16, 1)) * np.sqrt(1 / 16)).astype(np.float32)
    model.params["b3"] = np.zeros(1, dtype=np.float32)
    model.fit_normalizer(features[train_indices])
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.params.items()}
    step = 0
    max_steps = step_features.shape[2]
    for epoch in range(epochs):
        order = rng.permutation(train_indices)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = model.normalize(features[indices])
            n = len(indices)
            flat = batch.reshape(-1, batch.shape[-1])
            z1 = flat @ model.params["W1"] + model.params["b1"]
            hidden = np.maximum(z1, 0)
            if dropout:
                mask = (rng.random(hidden.shape) >= dropout).astype(np.float32) / (1 - dropout)
                dropped = hidden * mask
            else:
                mask = None
                dropped = hidden
            scores = (dropped @ model.params["W2"] + model.params["b2"]).reshape(n, 4)
            targets = labels[indices].astype(np.float32)
            targets /= targets.sum(axis=1, keepdims=True)
            ds = (softmax(scores) - targets) / n
            gradients = {}
            gradients["W2"] = dropped.T @ ds.reshape(-1, 1) + weight_decay * model.params["W2"]
            gradients["b2"] = ds.sum().reshape(1)
            dh = ds.reshape(-1, 1) @ model.params["W2"].T
            batch_steps = step_features[indices].reshape(-1, 16)
            repeated = np.repeat(dropped, max_steps, axis=0)
            logits = (repeated @ model.params["W3"] + batch_steps @ model.params["U3"] + model.params["b3"]).reshape(-1)
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            batch_labels = step_labels[indices].reshape(-1)
            batch_mask = step_mask[indices].reshape(-1)
            ds3 = alpha * (probabilities - batch_labels) * batch_mask / max(batch_mask.sum(), 1)
            gradients["W3"] = repeated.T @ ds3.reshape(-1, 1) + weight_decay * model.params["W3"]
            gradients["U3"] = batch_steps.T @ ds3.reshape(-1, 1) + weight_decay * model.params["U3"]
            gradients["b3"] = ds3.sum().reshape(1)
            dh += (ds3.reshape(-1, 1) @ model.params["W3"].T).reshape(n * 4, max_steps, hidden_dim).sum(axis=1)
            if mask is not None:
                dh *= mask
            dh[z1 <= 0] = 0
            gradients["W1"] = flat.T @ dh + weight_decay * model.params["W1"]
            gradients["b1"] = dh.sum(axis=0)
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
