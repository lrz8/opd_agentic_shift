from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from opd_agentic_shift.envs.encoder import obs_dim
from opd_agentic_shift.envs.troubleshooting_env import ACTIONS


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_dim: int = 96, action_dim: int | None = None):
        super().__init__()
        input_dim = obs_dim() if input_dim is None else input_dim
        action_dim = len(ACTIONS) if action_dim is None else action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act(self, obs_vec, deterministic: bool = False, temperature: float = 1.0):
        x = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x).squeeze(0) / max(temperature, 1e-6)
        if deterministic:
            action = int(torch.argmax(logits).item())
            logprob = torch.log_softmax(logits, dim=-1)[action].item()
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()
            action = int(a.item())
            logprob = float(dist.log_prob(a).item())
        return action, logprob


def save_policy(policy: MLPPolicy, path: str) -> None:
    torch.save({"state_dict": policy.state_dict()}, path)


def load_policy(path: str, map_location: str = "cpu") -> MLPPolicy:
    ckpt = torch.load(path, map_location=map_location)
    policy = MLPPolicy()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy
