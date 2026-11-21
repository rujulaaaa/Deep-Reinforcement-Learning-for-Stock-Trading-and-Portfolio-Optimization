"""Neural network building blocks for the DQN and DDPG agents."""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: list[int], out_dim: int, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q(s, ·) for all discrete actions; optional dueling head  Q = V + A − mean(A)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int], dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.body = mlp(obs_dim, hidden[:-1], hidden[-1])
            self.value = nn.Linear(hidden[-1], 1)
            self.adv = nn.Linear(hidden[-1], n_actions)
        else:
            self.net = mlp(obs_dim, hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)
        h = torch.relu(self.body(x))
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


class Actor(nn.Module):
    """Deterministic policy μ(s) → portfolio weights on the simplex (softmax of logits)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, action_dim, layer_norm=True)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)  # start near equal weights

    def logits(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(s), dim=-1)


class Critic(nn.Module):
    """Q(s, a) for continuous weight vectors a."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden, 1, layer_norm=True)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
