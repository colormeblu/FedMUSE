from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class OpenCLIPEncoder(nn.Module):
    """OpenCLIP image encoder wrapper.

    Goal: expose the same interface as `ResNetEncoder` used by the scaffold:
        forward(x) -> (feat: [B, D], feat_map: [B, D, H, W])

    Notes
    -----
    - We use `open_clip_torch` (import name: `open_clip`).
    - For ViT backbones, we try to recover patch-token features and reshape them
      into a spatial map. If that fails, we fall back to a 1x1 feature map.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "open_clip is not installed. Install with: pip install open_clip_torch"
            ) from e

        self._open_clip = open_clip
        self.model_name = model_name
        self.pretrained = pretrained

        # Create CLIP model (image+text); we'll only use the visual pathway.
        # If `pretrained` is empty/none, we initialize weights randomly (useful for offline dev).
        p = (pretrained or "").strip().lower()
        if p in ("", "none", "null", "false"):
            # open_clip expects a string; empty disables pretrained loading.
            self.clip = open_clip.create_model(model_name, pretrained="")
        else:
            self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = self.clip.visual

        # Feature dim (after projection) for most CLIP models.
        self.feat_dim = int(
            getattr(self.visual, "output_dim", 0)
            or getattr(self.clip, "embed_dim", 0)
            or 0
        )

        # Fallback dim inference if attributes are missing.
        if self.feat_dim <= 0:
            with torch.no_grad():
                dev = device or torch.device("cpu")
                dummy = torch.zeros(1, 3, 224, 224, device=dev)
                self.clip = self.clip.to(dev)
                out = self.clip.encode_image(dummy)
                self.feat_dim = int(out.shape[-1])

        if device is not None:
            self.clip = self.clip.to(device)

        # We do NOT normalize features here; leave that to losses / heads.

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Try to extract a spatial map. If it fails, return 1x1 map.
        feat, fmap = self._try_encode_with_map(x)
        return feat, fmap

    def _try_encode_with_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Preferred: ViT path (patch tokens -> spatial map)
        try:
            feat, fmap = self._encode_vit_with_map(x)
            return feat, fmap
        except Exception:
            pass

        # 2) Fallback: just pooled feature + 1x1 map
        feat = self.clip.encode_image(x)
        fmap = feat[:, :, None, None]
        return feat, fmap

    def _encode_vit_with_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Best-effort extraction of patch-token feature map for ViT CLIP models."""

        v = self.visual

        # Some OpenCLIP visual modules expose forward_features.
        if hasattr(v, "forward_features"):
            out = v.forward_features(x)
            # Possible shapes:
            # - [B, N, D] tokens
            # - [B, D, H, W] feature map
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                # Map already.
                fmap = out
                # pool
                feat = fmap.mean(dim=(2, 3))
                return feat, fmap
            if isinstance(out, torch.Tensor) and out.ndim == 3:
                tokens = out
                return self._tokens_to_feat_map(tokens)

        # OpenAI-style VisualTransformer path (common in open_clip)
        # We intentionally implement a minimal forward here.
        if not (hasattr(v, "conv1") and hasattr(v, "transformer")):
            raise RuntimeError("Visual module does not look like a ViT")

        # ---- Patchify ----
        x_ = v.conv1(x)  # [B, width, grid, grid]
        B, width, grid_h, grid_w = x_.shape
        x_ = x_.reshape(B, width, grid_h * grid_w).permute(0, 2, 1)  # [B, N, width]

        # ---- Add class token ----
        if hasattr(v, "class_embedding"):
            cls = v.class_embedding.to(x_.dtype)
            cls = cls + torch.zeros(B, 1, cls.shape[-1], dtype=x_.dtype, device=x_.device)
            x_ = torch.cat([cls, x_], dim=1)  # [B, 1+N, width]

        # ---- Positional embedding ----
        if hasattr(v, "positional_embedding"):
            pe = v.positional_embedding.to(x_.dtype)
            x_ = x_ + pe

        # ---- Pre-LN ----
        if hasattr(v, "ln_pre"):
            x_ = v.ln_pre(x_)

        # ---- Transformer ----
        x_ = x_.permute(1, 0, 2)  # [L, B, D]
        x_ = v.transformer(x_)
        x_ = x_.permute(1, 0, 2)  # [B, L, D]

        # ---- Post-LN + projection to embed_dim ----
        if hasattr(v, "ln_post"):
            x_ = v.ln_post(x_)
        if getattr(v, "proj", None) is not None:
            x_ = x_ @ v.proj

        # pooled feature: class token if present else mean
        if x_.shape[1] >= 2:
            feat = x_[:, 0, :]
            tokens = x_[:, 1:, :]
        else:
            feat = x_.mean(dim=1)
            tokens = x_

        # tokens -> fmap
        n = tokens.shape[1]
        g = int(math.sqrt(n))
        if g * g != n:
            # Can't reshape nicely; fall back.
            fmap = feat[:, :, None, None]
            return feat, fmap
        fmap = tokens[:, : g * g, :].permute(0, 2, 1).contiguous().reshape(B, -1, g, g)
        return feat, fmap

    def _tokens_to_feat_map(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, N, D] or [B, 1+N, D]
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B, N, D]")
        B, L, D = tokens.shape
        if L >= 2:
            feat = tokens[:, 0, :]
            patch = tokens[:, 1:, :]
        else:
            feat = tokens.mean(dim=1)
            patch = tokens
        n = patch.shape[1]
        g = int(math.sqrt(n))
        if g * g != n:
            fmap = feat[:, :, None, None]
        else:
            fmap = patch[:, : g * g, :].permute(0, 2, 1).contiguous().reshape(B, D, g, g)
        return feat, fmap


class OpenCLIPClassifier(nn.Module):
    """A simple linear classifier on top of OpenCLIP image features.

    It matches the `(logits, feat, feat_map)` interface used across the scaffold.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = OpenCLIPEncoder(model_name=model_name, pretrained=pretrained, device=device)
        self.head = nn.Linear(int(self.encoder.feat_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, fmap = self.encoder(x)
        logits = self.head(feat)
        return logits, feat, fmap


def build_encoder_from_cfg(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, int, str]:
    """Factory used by runners.

    Returns (encoder, feat_dim, transforms_name)
    """
    model_cfg = cfg.get("model", {}) or {}
    algo_cfg = cfg.get("algorithm", {}) or {}

    backbone = str(model_cfg.get("backbone", algo_cfg.get("backbone", "resnet18"))).lower()
    if backbone in ("open_clip", "openclip", "clip"):
        ocfg = (model_cfg.get("open_clip", {}) or {})
        name = str(ocfg.get("name", "ViT-B-32"))
        oc_pretrained = str(ocfg.get("pretrained", "laion2b_s34b_b79k"))
        enc = OpenCLIPEncoder(model_name=name, pretrained=oc_pretrained, device=device)
        return enc, int(enc.feat_dim), "clip"

    # Default: ResNet
    from fedmuse.models.resnet import ResNetEncoder

    b = str(model_cfg.get("backbone", algo_cfg.get("backbone", "resnet18")))
    enc = ResNetEncoder(backbone=b, pretrained=True).to(device)
    return enc, int(enc.feat_dim), "default"


def build_classifier_from_cfg(
    cfg: Dict[str, Any],
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
) -> Tuple[nn.Module, str]:
    """Factory for supervised backbones used by FedAvg baseline."""
    model_cfg = cfg.get("model", {}) or {}
    algo_cfg = cfg.get("algorithm", {}) or {}
    backbone = str(model_cfg.get("backbone", algo_cfg.get("backbone", "resnet18"))).lower()
    if backbone in ("open_clip", "openclip", "clip"):
        ocfg = (model_cfg.get("open_clip", {}) or {})
        name = str(ocfg.get("name", "ViT-B-32"))
        oc_pretrained = str(ocfg.get("pretrained", "laion2b_s34b_b79k"))
        # Note: OpenCLIP "pretrained" strings typically trigger weight downloads.
        # For federated clients we usually rebuild the same architecture and load
        # weights from the global model, so you can pass pretrained=False/"none".
        pre = oc_pretrained
        if isinstance(pretrained, bool) and not pretrained:
            pre = "none"
        m = OpenCLIPClassifier(
            num_classes=num_classes,
            model_name=name,
            pretrained=str(pre),
            device=device,
        ).to(device)
        return m, "clip"

    from fedmuse.models.resnet import ResNetClassifier

    b = str(model_cfg.get("backbone", algo_cfg.get("backbone", "resnet18")))
    m = ResNetClassifier(backbone=b, num_classes=int(num_classes), pretrained=bool(pretrained)).to(device)
    return m, "default"
