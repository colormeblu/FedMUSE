from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PromptMode = Literal["none", "global_only", "local_only", "joint"]


@dataclass
class PromptLayoutConfig:
    prompt_len: int = 4
    prompt_layers: int = 4
    start_layer: Optional[int] = None


class HierarchicalDualPrompt(nn.Module):
    """Hierarchical dual-stream prompts: shared(global) + local(expert)."""

    def __init__(
        self,
        width: int,
        num_layers: int,
        prompt_len: int = 4,
        prompt_layers: int = 4,
        start_layer: Optional[int] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.num_layers = int(num_layers)
        self.prompt_len = max(1, int(prompt_len))

        k = max(1, min(int(prompt_layers), self.num_layers))
        if start_layer is None:
            start = max(0, self.num_layers - k)
        else:
            start = max(0, min(int(start_layer), self.num_layers - 1))
        end = min(self.num_layers, start + k)

        active = torch.zeros(self.num_layers, dtype=torch.bool)
        active[start:end] = True
        self.register_buffer("active_mask", active, persistent=True)

        self.global_prompt = nn.Parameter(
            torch.randn(self.num_layers, self.prompt_len, self.width, dtype=torch.float32) * float(init_std)
        )
        self.local_prompt = nn.Parameter(
            torch.randn(self.num_layers, self.prompt_len, self.width, dtype=torch.float32) * float(init_std)
        )

        # Gate logits. Sigmoid(gate) is used as the actual scale.
        self.global_alpha = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        self.local_alpha = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))

    @staticmethod
    def _effective_from_state_dict(state: Dict[str, torch.Tensor], prompt_key: str, alpha_key: str) -> torch.Tensor:
        if prompt_key not in state or alpha_key not in state:
            raise KeyError(f"Missing keys for effective prompt: {prompt_key}, {alpha_key}")
        p = state[prompt_key]
        a = torch.sigmoid(state[alpha_key]).view(-1, 1, 1)
        if p.ndim != 3:
            raise ValueError(f"Prompt tensor must be 3D [L,P,D], got shape={tuple(p.shape)}")
        return a * p

    def effective_global_prompt(self) -> torch.Tensor:
        return torch.sigmoid(self.global_alpha).view(-1, 1, 1) * self.global_prompt

    def effective_local_prompt(self) -> torch.Tensor:
        return torch.sigmoid(self.local_alpha).view(-1, 1, 1) * self.local_prompt

    def layer_prompt(
        self,
        layer_idx: int,
        mode: PromptMode = "joint",
        external_prompt: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        li = int(layer_idx)
        if li < 0 or li >= self.num_layers:
            return None
        if not bool(self.active_mask[li].item()):
            return None

        if external_prompt is not None:
            if external_prompt.ndim != 3 or tuple(external_prompt.shape) != tuple(self.global_prompt.shape):
                raise ValueError(
                    "external_prompt must have shape "
                    f"{tuple(self.global_prompt.shape)}, got {tuple(external_prompt.shape)}"
                )
            return external_prompt[li].mean(dim=0, keepdim=True)

        if mode == "none":
            return None
        out: Optional[torch.Tensor] = None
        if mode in ("global_only", "joint"):
            g = torch.sigmoid(self.global_alpha[li]) * self.global_prompt[li].mean(dim=0, keepdim=True)
            out = g if out is None else out + g
        if mode in ("local_only", "joint"):
            l = torch.sigmoid(self.local_alpha[li]) * self.local_prompt[li].mean(dim=0, keepdim=True)
            out = l if out is None else out + l
        return out

    def global_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {
            "global_prompt": self.global_prompt.detach().cpu().clone(),
            "global_alpha": self.global_alpha.detach().cpu().clone(),
        }

    def local_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {
            "local_prompt": self.local_prompt.detach().cpu().clone(),
            "local_alpha": self.local_alpha.detach().cpu().clone(),
        }

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        out = self.global_state_dict_cpu()
        out.update(self.local_state_dict_cpu())
        return out

    def _copy_tensor_(self, param: nn.Parameter, src: torch.Tensor, name: str) -> None:
        src_t = src.to(device=param.device, dtype=param.dtype)
        if tuple(src_t.shape) != tuple(param.shape):
            raise ValueError(f"{name} shape mismatch: expected {tuple(param.shape)}, got {tuple(src_t.shape)}")
        with torch.no_grad():
            param.copy_(src_t)

    def load_global_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._copy_tensor_(self.global_prompt, state["global_prompt"], "global_prompt")
        self._copy_tensor_(self.global_alpha, state["global_alpha"], "global_alpha")

    def load_local_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._copy_tensor_(self.local_prompt, state["local_prompt"], "local_prompt")
        self._copy_tensor_(self.local_alpha, state["local_alpha"], "local_alpha")

    def load_state_dict_split(self, state: Dict[str, torch.Tensor]) -> None:
        self.load_global_state_dict(state)
        self.load_local_state_dict(state)


class PromptedOpenCLIPVision(nn.Module):
    """OpenCLIP ViT visual encoder with hierarchical dual-stream prompts."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        prompt_len: int = 4,
        prompt_layers: int = 4,
        prompt_start_layer: Optional[int] = None,
    ) -> None:
        super().__init__()

        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError("open_clip_torch is required. Install via `pip install open_clip_torch`.") from e

        self.model_name = str(model_name)
        self.pretrained = str(pretrained)

        pre = self.pretrained.strip().lower()
        if pre in ("", "none", "null", "false"):
            self.clip = open_clip.create_model(self.model_name, pretrained="")
        else:
            self.clip, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)

        self.visual = self.clip.visual
        if not hasattr(self.visual, "transformer") or not hasattr(self.visual.transformer, "resblocks"):
            raise RuntimeError("FedMUSE currently supports ViT-style OpenCLIP visual encoders only.")

        self.num_layers = len(self.visual.transformer.resblocks)
        if hasattr(self.visual, "conv1") and hasattr(self.visual.conv1, "weight"):
            width = int(self.visual.conv1.weight.shape[0])
        else:
            width = int(getattr(self.visual, "width", 768))
        self.width = width

        self.prompt = HierarchicalDualPrompt(
            width=self.width,
            num_layers=self.num_layers,
            prompt_len=int(prompt_len),
            prompt_layers=int(prompt_layers),
            start_layer=prompt_start_layer,
        )
        self.tokenizer = self._build_tokenizer(open_clip, self.model_name)

    @staticmethod
    def _build_tokenizer(open_clip_module, model_name: str):
        if hasattr(open_clip_module, "get_tokenizer"):
            return open_clip_module.get_tokenizer(model_name)
        if hasattr(open_clip_module, "tokenize"):
            tok = open_clip_module.tokenize

            def _tok(texts):
                return tok(list(texts))

            return _tok
        tok_mod = getattr(open_clip_module, "tokenizer", None)
        if tok_mod is not None and hasattr(tok_mod, "tokenize"):
            tok = tok_mod.tokenize

            def _tok(texts):
                return tok(list(texts))

            return _tok
        raise RuntimeError("Cannot find a tokenizer API in open_clip module.")

    def freeze_backbone(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad_(False)
        for p in self.prompt.parameters():
            p.requires_grad_(True)

    def encode_text(self, texts: Sequence[str], device: torch.device, normalize: bool = True) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(device)
        with torch.no_grad():
            txt = self.clip.encode_text(tokens).float()
            if normalize:
                txt = F.normalize(txt, dim=-1)
        return txt

    def get_logit_scale(self) -> float:
        if hasattr(self.clip, "logit_scale"):
            with torch.no_grad():
                return float(self.clip.logit_scale.exp().item())
        return 100.0

    def encode_image(
        self,
        x: torch.Tensor,
        prompt_mode: PromptMode = "joint",
        normalize: bool = True,
        external_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self._forward_visual(x, prompt_mode=prompt_mode, external_prompt=external_prompt).float()
        if normalize:
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        prompt_mode: PromptMode = "joint",
        normalize: bool = True,
        external_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode_image(
            x,
            prompt_mode=prompt_mode,
            normalize=normalize,
            external_prompt=external_prompt,
        )

    def _forward_visual(
        self,
        x: torch.Tensor,
        prompt_mode: PromptMode = "joint",
        external_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = self.visual
        ext = None
        if external_prompt is not None:
            ext = external_prompt.to(device=x.device, dtype=x.dtype)

        x = v.conv1(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)

        cls = v.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls + torch.zeros(batch_size, 1, cls.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        if hasattr(v, "positional_embedding"):
            x = x + v.positional_embedding.to(dtype=x.dtype, device=x.device)
        if hasattr(v, "patch_dropout"):
            x = v.patch_dropout(x)
        if hasattr(v, "ln_pre"):
            x = v.ln_pre(x)

        x = x.permute(1, 0, 2)
        for layer_idx, block in enumerate(v.transformer.resblocks):
            x = block(x)
            p = self.prompt.layer_prompt(layer_idx, mode=prompt_mode, external_prompt=ext)
            if p is not None:
                x = x + p.to(dtype=x.dtype, device=x.device).view(1, 1, -1)

        x = x.permute(1, 0, 2)
        if hasattr(v, "ln_post"):
            x = v.ln_post(x)

        pool_type = getattr(v, "pool_type", "tok")
        if pool_type == "avg":
            feat = x[:, 1:, :].mean(dim=1)
        else:
            feat = x[:, 0, :]
        if getattr(v, "proj", None) is not None:
            feat = feat @ v.proj
        return feat

    @staticmethod
    def effective_prompt_from_state(state: Dict[str, torch.Tensor], stream: Literal["global", "local"]) -> torch.Tensor:
        if stream == "global":
            return HierarchicalDualPrompt._effective_from_state_dict(state, "global_prompt", "global_alpha")
        return HierarchicalDualPrompt._effective_from_state_dict(state, "local_prompt", "local_alpha")

    def global_prompt_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return self.prompt.global_state_dict_cpu()

    def local_prompt_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return self.prompt.local_state_dict_cpu()

    def prompt_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return self.prompt.state_dict_cpu()

    def load_global_prompt_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompt.load_global_state_dict(state)

    def load_local_prompt_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompt.load_local_state_dict(state)

    def load_prompt_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompt.load_state_dict_split(state)

    def global_prompt_tensor(self) -> torch.Tensor:
        return self.prompt.global_prompt

    def local_prompt_tensor(self) -> torch.Tensor:
        return self.prompt.local_prompt

    def effective_global_prompt(self) -> torch.Tensor:
        return self.prompt.effective_global_prompt()

    def effective_local_prompt(self) -> torch.Tensor:
        return self.prompt.effective_local_prompt()


# Backward-compatible aliases for older imports.
MultiScaleVisualPrompt = HierarchicalDualPrompt
