from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

# We try to use torchvision ResNets if available. If torchvision is unavailable
# (or broken due to binary mismatch), we fall back to a small CNN so the scaffold
# remains runnable everywhere.

_RESNETS = {
    "resnet18": ("resnet18", 512),
    "resnet50": ("resnet50", 2048),
}

def _try_build_torchvision_resnet(name: str, pretrained: bool):
    try:
        from torchvision import models  # lazy import
    except Exception as e:
        return None, None

    ctor_name, feat_dim = _RESNETS.get(name, ("resnet18", 512))
    ctor = getattr(models, ctor_name, None)
    if ctor is None:
        return None, None

    # Torchvision API changed from `pretrained=` to `weights=`.
    try:
        m = ctor(weights="DEFAULT" if pretrained else None)
    except Exception:
        try:
            m = ctor(pretrained=pretrained)
        except Exception:
            return None, None

    # remove classification head; keep conv trunk
    trunk = nn.Sequential(*list(m.children())[:-2])  # up to layer4 output (B,C,H,W)
    return trunk, feat_dim

class SimpleCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        backbone = backbone.lower()
        trunk, feat_dim = _try_build_torchvision_resnet(backbone, pretrained=pretrained)
        if trunk is None:
            # fallback
            self.trunk = SimpleCNN(out_dim=256)
            self.feat_dim = 256
        else:
            self.trunk = trunk
            self.feat_dim = int(feat_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.trunk(x)  # [B, C, H, W]
        feat = self.pool(feat_map).flatten(1)  # [B, C]
        return feat, feat_map

class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)

class ResNetClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(backbone, pretrained=pretrained)
        self.head = LinearHead(self.encoder.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, feat_map = self.encoder(x)
        logits = self.head(feat)
        return logits, feat, feat_map
