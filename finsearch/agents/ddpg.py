"""
Deep Deterministic Policy Gradient agent for continuous portfolio allocation.

Actor  μ(s)   → softmax over (cash + N stocks): weights are valid by construction
Critic Q(s,a) → expected discounted shaped return of holding weights a in state s

Exploration adds Gaussian noise to the actor's *pre-softmax logits* (with a
linearly decaying σ), which keeps the noisy action on the simplex — adding
noise to the weights directly would require re-projection.
Target networks are updated with Polyak averaging θ⁻ ← τθ + (1−τ)θ⁻.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict, device: str = "cpu", seed: int = 0):
        self.c = cfg
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(obs_dim, action_dim, cfg["actor_hidden"]).to(self.device)
        self.critic = Critic(obs_dim, action_dim, cfg["critic_hidden"]).to(self.device)
        self.actor_t = copy.deepcopy(self.actor).eval()
        self.critic_t = copy.deepcopy(self.critic).eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"],
                                           weight_decay=cfg.get("weight_decay", 0.0))
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, (action_dim,), np.float32, seed=seed)
        self.steps = 0

    def sigma(self) -> float:
        frac = min(1.0, self.steps / max(1, self.c["noise_decay_steps"]))
        return self.c["noise_sigma"] + frac * (self.c["noise_sigma_end"] - self.c["noise_sigma"])

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = False) -> np.ndarray:
        s = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        self.actor.eval()
        logits = self.actor.logits(s).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore:
            logits = logits + self.rng.normal(0.0, self.sigma(), size=logits.shape)
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, terminal) -> dict | None:
        self.buffer.add(obs, action, reward, next_obs, terminal)
        self.steps += 1
        if self.steps >= self.c["learning_starts"] and self.steps % self.c.get("train_freq", 1) == 0:
            return self.update()
        return None

    def update(self) -> dict:
        s, a, r, s2, d = self.buffer.sample(self.c["batch_size"], self.device)
        with torch.no_grad():
            target = r + self.c["gamma"] * (1.0 - d) * self.critic_t(s2, self.actor_t(s2))
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.c["max_grad_norm"])
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.c["max_grad_norm"])
        self.actor_opt.step()

        tau = self.c["tau"]
        with torch.no_grad():
            for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "q_mean": q.mean().item()}

    def state_dict(self) -> dict:
        return {"actor": copy.deepcopy(self.actor.state_dict()),
                "critic": copy.deepcopy(self.critic.state_dict())}

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_t.load_state_dict(sd["actor"])
        self.critic_t.load_state_dict(sd["critic"])
