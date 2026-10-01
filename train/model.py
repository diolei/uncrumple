"""MLP core: tanh or sine (SIREN-style) hidden layers, linear output.
Uniform Xavier-style init with zero biases; SIREN nets widen the first
layer by w0/sqrt(fan_in) (w0=30) so high-dimensional normalized inputs
still span several sine periods instead of saturating."""

import math

import torch
from torch import nn

SIREN_W0 = 30.0


class Mlp(nn.Module):
    def __init__(self, sizes: list, seed: int, act: str = "tanh"):
        super().__init__()
        self.sizes = list(sizes)
        assert act in ("tanh", "sine"), act
        self.act = act
        gen = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList()
        for li, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = nn.Linear(in_dim, out_dim)
            if act == "sine" and li == 0:
                s = math.sqrt(6.0 / in_dim) * SIREN_W0 / math.sqrt(in_dim)
            elif act == "sine":
                s = math.sqrt(6.0 / in_dim)
            else:
                s = math.sqrt(2.0 / (in_dim + out_dim))
            with torch.no_grad():
                layer.weight.uniform_(-s, s, generator=gen)
                layer.bias.zero_()
            self.layers.append(layer)

    def _nonlin(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) if self.act == "sine" else torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self._nonlin(x)
        return x


def snapshot(net: Mlp) -> dict:
    """Plain-python snapshot: float64 nested lists, [out][in] rows."""
    sd = {k: v.detach().to(torch.float64) for k, v in net.state_dict().items()}
    weights, biases = [], []
    for i in range(len(net.sizes) - 1):
        weights.append(sd[f"layers.{i}.weight"].tolist())
        biases.append(sd[f"layers.{i}.bias"].tolist())
    return {"sizes": list(net.sizes), "act": net.act,
            "weights": weights, "biases": biases}


def nets_from_snapshots(stages: list) -> list:
    """Rebuild frozen nets from snapshots (continuation training)."""
    import numpy as np

    nets = []
    for st in stages:
        net = Mlp(st["sizes"], 0, st.get("act", "tanh"))
        sd = {}
        for i in range(len(st["sizes"]) - 1):
            sd[f"layers.{i}.weight"] = torch.tensor(
                np.asarray(st["weights"][i]), dtype=torch.float32
            )
            sd[f"layers.{i}.bias"] = torch.tensor(
                np.asarray(st["biases"][i]), dtype=torch.float32
            )
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return nets
