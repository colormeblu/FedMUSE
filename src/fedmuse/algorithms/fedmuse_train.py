from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedmuse.utils.metrics import AverageMeter, accuracy_top1


def _progress_iter(loader: DataLoader, progress: Optional[Dict[str, Any]]):
    if not progress or not progress.get("enabled", False):
        return None, loader
    try:
        from tqdm import tqdm
    except Exception:
        return None, loader
    pbar = tqdm(
        loader,
        desc=progress.get("desc"),
        leave=bool(progress.get("leave", False)),
        dynamic_ncols=bool(progress.get("dynamic_ncols", True)),
    )
    return pbar, pbar


def _model_core(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _model_encode_image(
    model,
    x: torch.Tensor,
    prompt_mode: str,
    normalize: bool = True,
    external_prompt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # DataParallel scatters Tensor kwargs by dim=0, which breaks shared prompts
    # shaped as [L, P, D]. Use core model directly when external prompt is given.
    if isinstance(model, torch.nn.DataParallel):
        if external_prompt is not None:
            core = _model_core(model)
            return core.encode_image(
                x,
                prompt_mode=prompt_mode,
                normalize=normalize,
                external_prompt=external_prompt,
            )
        # Use keyword x for safer argument binding under DataParallel.
        return model(
            x=x,
            prompt_mode=prompt_mode,
            normalize=normalize,
            external_prompt=None,
        )
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model(
            x=x,
            prompt_mode=prompt_mode,
            normalize=normalize,
            external_prompt=external_prompt,
        )
    return model.encode_image(
        x,
        prompt_mode=prompt_mode,
        normalize=normalize,
        external_prompt=external_prompt,
    )


def clip_classification_logits(
    image_features: torch.Tensor,
    class_text_features: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    return float(logit_scale) * image_features @ class_text_features.t()


class StyleMappingNetwork(nn.Module):
    """Map text-style embeddings to AdaIN style statistics."""

    def __init__(self, text_dim: int, feature_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        h = max(64, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(text_dim), h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 2 * int(feature_dim)),
        )

    def forward(self, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(text_features)
        mu, raw_sigma = out.chunk(2, dim=-1)
        sigma = F.softplus(raw_sigma) + 1e-4
        return mu, sigma

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}


def adain_feature_transform(
    features: torch.Tensor,
    style_mu: torch.Tensor,
    style_sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """AdaIN on feature vectors.

    features: [B, D]
    style_mu/style_sigma: [K, D] or [B, K, D]
    returns: [B, K, D]
    """
    if features.ndim != 2:
        raise ValueError("features must be [B,D]")
    feat_mean = features.mean(dim=-1, keepdim=True)
    feat_std = features.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
    normed = (features - feat_mean) / feat_std

    if style_mu.ndim == 2:
        mu = style_mu.unsqueeze(0)
        sigma = style_sigma.unsqueeze(0)
    elif style_mu.ndim == 3:
        mu = style_mu
        sigma = style_sigma
    else:
        raise ValueError("style_mu must be [K,D] or [B,K,D]")
    return sigma * normed.unsqueeze(1) + mu


def orthogonal_disentanglement_loss(global_prompts: torch.Tensor, local_prompts: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of cosine-similarity matrix between global/local prompts."""
    if global_prompts.ndim != 3 or local_prompts.ndim != 3:
        raise ValueError("global/local prompts must be [L,P,D]")
    if tuple(global_prompts.shape) != tuple(local_prompts.shape):
        raise ValueError(
            f"global/local shape mismatch: {tuple(global_prompts.shape)} vs {tuple(local_prompts.shape)}"
        )
    g = F.normalize(global_prompts.reshape(-1, global_prompts.size(-1)), dim=-1)
    l = F.normalize(local_prompts.reshape(-1, local_prompts.size(-1)), dim=-1)
    sim = g @ l.t()
    return sim.pow(2).mean()


def global_local_contrastive_loss(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor,
    temperature: float,
    prototype_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kept for backward-compatible tests/ablations."""
    if image_features.ndim != 2 or global_prototypes.ndim != 2:
        raise ValueError("image_features and global_prototypes must both be 2D tensors")
    num_classes = int(global_prototypes.size(0))
    if num_classes <= 0:
        return torch.tensor(0.0, device=image_features.device)
    if prototype_mask is None:
        mask = torch.ones(num_classes, dtype=torch.bool, device=image_features.device)
    else:
        mask = prototype_mask.to(device=image_features.device, dtype=torch.bool)
        if mask.ndim != 1 or int(mask.numel()) != num_classes:
            raise ValueError("prototype_mask must be 1D with length == num_classes")
    if not bool(mask.any()):
        return torch.tensor(0.0, device=image_features.device)
    tau = max(float(temperature), 1e-6)
    logits = (image_features @ global_prototypes.t()) / tau
    logits = logits.masked_fill(~mask.view(1, -1), -1e4)
    targets = labels.clone().to(dtype=torch.long)
    valid = (targets >= 0) & (targets < num_classes)
    if bool(valid.any()):
        safe_targets = targets.clamp(min=0, max=num_classes - 1)
        has_proto = mask[safe_targets]
        valid = valid & has_proto
    targets[~valid] = -100
    if not bool(valid.any()):
        return torch.tensor(0.0, device=image_features.device)
    return F.cross_entropy(logits, targets, ignore_index=-100)


def _semantic_consistency_loss(
    base_features: torch.Tensor,
    base_logits: torch.Tensor,
    class_text_features: torch.Tensor,
    style_text_features: Optional[torch.Tensor],
    style_mapper: StyleMappingNetwork,
    logit_scale: float,
) -> torch.Tensor:
    if style_text_features is None or style_text_features.numel() == 0:
        return torch.zeros((), device=base_features.device, dtype=base_features.dtype)

    with torch.no_grad():
        teacher_prob = F.softmax(base_logits, dim=-1)

    style_mu, style_sigma = style_mapper(style_text_features)
    style_feat = adain_feature_transform(base_features, style_mu, style_sigma)
    style_feat = F.normalize(style_feat, dim=-1)
    style_logits = float(logit_scale) * torch.einsum("bkd,cd->bkc", style_feat, class_text_features)

    teacher = teacher_prob.unsqueeze(1).expand_as(style_logits)
    kl = F.kl_div(F.log_softmax(style_logits, dim=-1), teacher, reduction="none")
    return kl.sum(dim=-1).mean()


def train_fedmuse_epoch(
    model,
    style_mapper: StyleMappingNetwork,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    style_text_features: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_sem: float,
    lambda_orth: float,
    logit_scale: float,
    use_semantic_hallucination: bool = True,
    use_orth_loss: bool = True,
    progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    core = _model_core(model)
    core.prompt.train()
    style_mapper.train()

    ce_meter = AverageMeter()
    sem_meter = AverageMeter()
    orth_meter = AverageMeter()
    total_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar, loader_iter = _progress_iter(loader, progress)
    for x, y in loader_iter:
        x = x.to(device)
        y = y.to(device)

        feat = _model_encode_image(model, x, prompt_mode="joint", normalize=False)
        feat_n = F.normalize(feat, dim=-1)
        logits = clip_classification_logits(feat_n, class_text_features, logit_scale=logit_scale)
        ce = F.cross_entropy(logits, y)

        if bool(use_semantic_hallucination) and float(lambda_sem) > 0.0:
            sem = _semantic_consistency_loss(
                base_features=feat,
                base_logits=logits.detach(),
                class_text_features=class_text_features,
                style_text_features=style_text_features,
                style_mapper=style_mapper,
                logit_scale=logit_scale,
            )
        else:
            sem = torch.zeros((), device=device, dtype=ce.dtype)

        if bool(use_orth_loss) and float(lambda_orth) > 0.0:
            orth = orthogonal_disentanglement_loss(core.global_prompt_tensor(), core.local_prompt_tensor())
        else:
            orth = torch.zeros((), device=device, dtype=ce.dtype)

        total = ce + float(lambda_sem) * sem + float(lambda_orth) * orth

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        bsz = int(x.size(0))
        ce_meter.update(float(ce.item()), n=bsz)
        sem_meter.update(float(sem.item()), n=bsz)
        orth_meter.update(float(orth.item()), n=bsz)
        total_meter.update(float(total.item()), n=bsz)
        acc_meter.update(float(accuracy_top1(logits.detach(), y)), n=bsz)

    if pbar is not None:
        pbar.close()

    return {
        "ce_loss": float(ce_meter.avg),
        "sem_loss": float(sem_meter.avg),
        "orth_loss": float(orth_meter.avg),
        "total_loss": float(total_meter.avg),
        "acc": float(acc_meter.avg),
    }


@torch.no_grad()
def collect_feature_stats(
    model,
    loader: DataLoader,
    device: torch.device,
    prompt_mode: str = "joint",
    max_batches: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect diagonal Gaussian stats for Mahalanobis matching."""
    core = _model_core(model)
    core.prompt.eval()
    feats = []
    for batch_idx, (x, _) in enumerate(loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        x = x.to(device)
        f = _model_encode_image(model, x, prompt_mode=prompt_mode, normalize=False).float()
        feats.append(f.detach().cpu())

    if len(feats) == 0:
        raise RuntimeError("Cannot collect feature stats from an empty loader.")

    feat = torch.cat(feats, dim=0)
    mu = feat.mean(dim=0)
    var = feat.var(dim=0, unbiased=False).clamp_min(1e-5)
    return mu, var


def mahalanobis_distance_diag(
    feature: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    diff = feature - mean
    inv_var = 1.0 / (var + float(eps))
    d2 = torch.sum(diff * diff * inv_var, dim=-1)
    return torch.sqrt(d2.clamp_min(0.0))


def predictive_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    prob = F.softmax(logits, dim=-1)
    return -(prob * torch.log(prob.clamp_min(float(eps)))).sum(dim=-1)


def uncertainty_gate(entropy: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    return torch.sigmoid(float(beta) * entropy + float(gamma))


def fuse_expert_weights(
    distance_weights: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    w_dist = distance_weights / distance_weights.sum().clamp_min(float(eps))
    if router_logits is None:
        return w_dist
    w_router = torch.softmax(router_logits, dim=-1)
    logits = torch.log(w_dist.clamp_min(float(eps))) + torch.log(w_router.clamp_min(float(eps)))
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def eval_fedmuse_ua_ttaf(
    model,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    device: torch.device,
    logit_scale: float,
    global_prompt: torch.Tensor,
    expert_prompts: Sequence[torch.Tensor],
    expert_means: Sequence[torch.Tensor],
    expert_vars: Sequence[torch.Tensor],
    beta: float,
    gamma: float,
    router: Optional[nn.Module] = None,
    use_ua_ttaf: bool = True,
) -> Tuple[float, float, float]:
    core = _model_core(model)
    core.prompt.eval()
    if router is not None:
        router.eval()

    global_prompt = global_prompt.to(device=device, dtype=torch.float32)
    exp_prompts = [p.to(device=device, dtype=torch.float32) for p in expert_prompts]
    exp_means = [m.to(device=device, dtype=torch.float32) for m in expert_means]
    exp_vars = [v.to(device=device, dtype=torch.float32) for v in expert_vars]

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    alpha_meter = AverageMeter()

    n_experts = len(exp_prompts)
    if not (len(exp_means) == n_experts and len(exp_vars) == n_experts):
        raise ValueError("expert_prompts/expert_means/expert_vars length mismatch")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        bsz = int(x.size(0))

        for i in range(bsz):
            xi = x[i : i + 1]
            yi = y[i : i + 1]

            feat_g_raw = _model_encode_image(model, xi, prompt_mode="global_only", normalize=False)
            feat_g = F.normalize(feat_g_raw, dim=-1)
            logits_g = clip_classification_logits(feat_g, class_text_features, logit_scale=logit_scale)

            if not bool(use_ua_ttaf):
                alpha = torch.zeros(1, device=device, dtype=feat_g.dtype)
                final_prompt = global_prompt
            else:
                entropy = predictive_entropy(logits_g)
                alpha = uncertainty_gate(entropy, beta=beta, gamma=gamma).view(1)

                if n_experts > 0:
                    dists = []
                    fg = feat_g_raw.squeeze(0)
                    for m, v in zip(exp_means, exp_vars):
                        d = mahalanobis_distance_diag(fg, m, v)
                        dists.append(d)
                    d_t = torch.stack(dists, dim=0)
                    w_dist = torch.softmax(-d_t, dim=0)

                    router_logits = None
                    if router is not None:
                        router_logits = router(feat_g_raw).squeeze(0)
                    w = fuse_expert_weights(w_dist, router_logits=router_logits)

                    prompt_mix = torch.zeros_like(global_prompt)
                    for j in range(n_experts):
                        prompt_mix = prompt_mix + w[j] * exp_prompts[j]
                    final_prompt = (1.0 - alpha.item()) * global_prompt + alpha.item() * prompt_mix
                else:
                    final_prompt = global_prompt

            feat_f = _model_encode_image(
                model,
                xi,
                prompt_mode="none",
                normalize=True,
                external_prompt=final_prompt,
            )
            logits_f = clip_classification_logits(feat_f, class_text_features, logit_scale=logit_scale)
            loss = F.cross_entropy(logits_f, yi)

            loss_meter.update(float(loss.item()), n=1)
            acc_meter.update(float(accuracy_top1(logits_f, yi)), n=1)
            alpha_meter.update(float(alpha.item()), n=1)

    return float(loss_meter.avg), float(acc_meter.avg), float(alpha_meter.avg)
