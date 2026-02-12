from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def fedavg_state_dict(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Weighted average by sample count."""
    total = float(sum(n for _, n in updates))
    if total <= 0:
        raise ValueError("Total samples must be > 0")
    return weighted_average_state_dict([u[0] for u in updates], [float(u[1]) / total for u in updates])


def weighted_average_state_dict(
    states: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    if len(states) == 0:
        raise ValueError("states must be non-empty")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length")

    w = torch.tensor(list(weights), dtype=torch.float32)
    if float(w.sum().item()) <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / w.sum()
    ref_idx = int(torch.argmax(w).item())

    ref_keys = set(states[0].keys())
    for idx, sd in enumerate(states[1:], start=1):
        keys = set(sd.keys())
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            parts = [f"state_dict keys mismatch at index {idx}"]
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            raise ValueError("; ".join(parts))

    out: Dict[str, torch.Tensor] = {}
    for k, v0 in states[0].items():
        # Floating/complex tensors are safely averaged.
        if torch.is_floating_point(v0) or torch.is_complex(v0):
            acc = torch.zeros_like(v0, dtype=v0.dtype)
            for idx, sd in enumerate(states):
                wi = float(w[idx].item())
                acc = acc + sd[k].detach().to(dtype=v0.dtype, device=v0.device) * wi
            out[k] = acc
        else:
            # Non-floating buffers (e.g. BatchNorm num_batches_tracked) are not
            # mathematically averaged; copy from the highest-weight client.
            out[k] = states[ref_idx][k].detach().clone()
    return out


def semantic_softmax_weights(scores: Sequence[float], tau: float) -> List[float]:
    if len(scores) == 0:
        return []
    t = max(float(tau), 1e-6)
    s = torch.tensor(list(scores), dtype=torch.float32)

    # Avoid NaN/Inf from broken clients; push them to low confidence.
    finite = torch.isfinite(s)
    if not bool(finite.all()):
        if bool(finite.any()):
            floor = float(torch.min(s[finite]).item()) - 10.0
        else:
            floor = -10.0
        s = torch.where(finite, s, torch.full_like(s, floor))

    w = torch.softmax(s / t, dim=0)
    return [float(x.item()) for x in w]


def copy_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def aggregate_class_prototypes(
    client_prototypes: Sequence[torch.Tensor],
    client_counts: Sequence[torch.Tensor],
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-client class prototypes into global class prototypes."""
    if len(client_prototypes) == 0:
        raise ValueError("client_prototypes must be non-empty")
    if len(client_prototypes) != len(client_counts):
        raise ValueError("client_prototypes and client_counts must have the same length")

    ref_proto = client_prototypes[0]
    if ref_proto.ndim != 2:
        raise ValueError("Each prototype tensor must be 2D: [num_classes, feat_dim]")
    num_classes, feat_dim = int(ref_proto.size(0)), int(ref_proto.size(1))

    weighted_sum = torch.zeros(num_classes, feat_dim, dtype=torch.float32)
    total_count = torch.zeros(num_classes, dtype=torch.float32)

    for idx, (proto, counts) in enumerate(zip(client_prototypes, client_counts)):
        if proto.ndim != 2 or tuple(proto.shape) != (num_classes, feat_dim):
            raise ValueError(
                f"Prototype shape mismatch at index {idx}: "
                f"expected {(num_classes, feat_dim)}, got {tuple(proto.shape)}"
            )
        if counts.ndim != 1 or int(counts.numel()) != num_classes:
            raise ValueError(
                f"Count shape mismatch at index {idx}: expected ({num_classes},), got {tuple(counts.shape)}"
            )

        p = proto.detach().to(dtype=torch.float32, device="cpu")
        c = counts.detach().to(dtype=torch.float32, device="cpu").clamp(min=0.0)
        weighted_sum = weighted_sum + p * c.unsqueeze(1)
        total_count = total_count + c

    global_proto = torch.zeros_like(weighted_sum)
    valid_mask = total_count > 0
    if bool(valid_mask.any()):
        global_proto[valid_mask] = weighted_sum[valid_mask] / total_count[valid_mask].unsqueeze(1)
        if normalize:
            global_proto[valid_mask] = F.normalize(global_proto[valid_mask], dim=-1)

    return global_proto, valid_mask
