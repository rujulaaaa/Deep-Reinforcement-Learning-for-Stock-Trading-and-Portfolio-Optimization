"""Fixed-capacity circular experience replay buffer (numpy-backed)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_shape: tuple, action_dtype=np.float32,
                 seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminal: bool) -> None:
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(terminal)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.dones)

    def __len__(self) -> int:
        return self.size
