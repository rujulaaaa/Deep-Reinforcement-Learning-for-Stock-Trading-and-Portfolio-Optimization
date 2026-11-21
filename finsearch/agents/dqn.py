"""
Deep Q-Network agent for the discrete single-stock trading task.

Built on the mid-term CartPole implementation (replay buffer, Huber loss,
target network, gradient clipping), plus two extensions motivated by the
mid-term's observed limitations:

* Double DQN  — online net selects a', target net evaluates it; reduces the
  Q-value over-estimation that is especially harmful on noisy price data.
* Dueling head — separates state value from action advantage; most market
  states have similar value regardless of action, which this exploits.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, device: str = "cpu", seed: int = 0):
        self.c = cfg
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(obs_dim, n_actions, cfg["hidden"], cfg["dueling"]).to(self.device)
        self.q_target = copy.deepcopy(self.q).eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, (), np.int64, seed=seed)
        self.steps = 0
        self.updates = 0

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        frac = min(1.0, self.steps / max(1, self.c["eps_decay_steps"]))
        return self.c["eps_start"] + frac * (self.c["eps_end"] - self.c["eps_start"])

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def observe(self, obs, action, reward, next_obs, terminal) -> dict | None:
        self.buffer.add(obs, action, reward, next_obs, terminal)
        self.steps += 1
        if self.steps >= self.c["learning_starts"] and self.steps % self.c["train_freq"] == 0:
            return self.update()
        return None

    def update(self) -> dict:
        s, a, r, s2, d = self.buffer.sample(self.c["batch_size"], self.device)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.c["double"]:
                a2 = self.q(s2).argmax(dim=1, keepdim=True)
                q2 = self.q_target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.q_target(s2).max(dim=1).values
            target = r + self.c["gamma"] * (1.0 - d) * q2
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.c["max_grad_norm"])
        self.opt.step()
        self.updates += 1
        self._update_target()
        return {"loss": loss.item(), "q_mean": q.mean().item()}

    def _update_target(self) -> None:
        tau = self.c.get("tau", 0.0)
        if tau and tau > 0:
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)
        elif self.updates % self.c["target_update"] == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"q": copy.deepcopy(self.q.state_dict())}

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q"])
