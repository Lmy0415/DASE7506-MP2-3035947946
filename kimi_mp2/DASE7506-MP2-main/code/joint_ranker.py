import numpy as np
import torch
from torch import nn


class JointRanker(nn.Module):
    def __init__(self, input_dim=1057, hidden_dim=192, heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = torch.relu(self.projection(normalized))
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm(hidden + self.dropout(attended))
        return self.output(hidden).squeeze(-1)


def save_joint(model, path):
    arrays = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    arrays["config"] = np.array([model.input_dim, model.hidden_dim, model.heads], dtype=np.int64)
    np.savez(path, **arrays)


def load_joint(path):
    with np.load(path, allow_pickle=False) as asset:
        config = asset["config"]
        model = JointRanker(int(config[0]), int(config[1]), int(config[2]))
        arrays = {name: asset[name].copy() for name in asset.files if name != "config"}
    state = {name: torch.from_numpy(value) for name, value in arrays.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def joint_scores(model, features, batch_size=512):
    output = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size])
            output.append(model(batch).numpy())
    return np.concatenate(output)
