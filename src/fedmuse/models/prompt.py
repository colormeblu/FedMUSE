from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleStats(nn.Module):
    """Extract simple style stats from a conv feature map: channel mean + std."""
    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: [B, C, H, W]
        mu = feat_map.mean(dim=(2,3))
        var = feat_map.var(dim=(2,3), unbiased=False)
        sigma = torch.sqrt(var + 1e-6)
        return torch.cat([mu, sigma], dim=1)  # [B, 2C]

class PromptGenerator(nn.Module):
    """3-layer MLP: (2C) -> 512 -> 512 -> D (prompt vector)."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        return self.net(style)

class PromptedHead(nn.Module):
    """Add prompt vector to pooled feature before classification."""
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feat: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        return self.fc(feat + prompt)

def kl_sym(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    kl_pq = F.kl_div(p, q, reduction="batchmean")
    q_log = F.log_softmax(q_logits, dim=1)
    p_sm = F.softmax(p_logits, dim=1)
    kl_qp = F.kl_div(q_log, p_sm, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)
