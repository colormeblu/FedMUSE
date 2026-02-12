from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedmuse.models.prompt import StyleStats, PromptGenerator, PromptedHead, kl_sym
from fedmuse.utils.metrics import AverageMeter, accuracy_top1

@dataclass
class CausalPromptState:
    generator: PromptGenerator
    head: PromptedHead
    style: StyleStats

def build_causal_prompt_modules(feat_dim: int, num_classes: int, hidden: int = 512) -> CausalPromptState:
    style = StyleStats()
    gen = PromptGenerator(in_dim=2*feat_dim, out_dim=feat_dim, hidden=hidden)
    head = PromptedHead(feat_dim=feat_dim, num_classes=num_classes)
    return CausalPromptState(generator=gen, head=head, style=style)

def train_causal_prompt_epoch(
    encoder,  # frozen encoder that returns (feat, feat_map)
    modules: CausalPromptState,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    lambda_causal: float = 1.0,
    progress: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    encoder.eval()  # frozen
    modules.generator.train()
    modules.head.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = None
    loader_iter = loader
    if progress and progress.get("enabled", False):
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        if tqdm is not None:
            pbar = tqdm(
                loader,
                desc=progress.get("desc"),
                leave=bool(progress.get("leave", False)),
                dynamic_ncols=bool(progress.get("dynamic_ncols", True)),
            )
            loader_iter = pbar

    for x, y in loader_iter:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            feat, feat_map = encoder(x)  # [B,D], [B,C,H,W]

        style_stats = modules.style(feat_map)         # [B,2C] where C==D for resnet last conv channels after pool? (We use encoder.feat_dim)
        prompt = modules.generator(style_stats)
        logits = modules.head(feat, prompt)
        ce = F.cross_entropy(logits, y)

        inv = torch.tensor(0.0, device=device)
        if lambda_causal > 0:
            # counterfactual style: shuffle style stats in batch (approx. using S_{x'}).
            idx = torch.randperm(style_stats.size(0), device=device)
            style_cf = style_stats[idx]
            prompt_cf = modules.generator(style_cf)
            logits_cf = modules.head(feat, prompt_cf)
            inv = kl_sym(logits, logits_cf)
        loss = ce + float(lambda_causal) * inv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits.detach(), y), n=x.size(0))

    if pbar is not None:
        pbar.close()

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def eval_causal_prompt(
    encoder,
    modules: CausalPromptState,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    encoder.eval()
    modules.generator.eval()
    modules.head.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        feat, feat_map = encoder(x)
        style_stats = modules.style(feat_map)
        prompt = modules.generator(style_stats)
        logits = modules.head(feat, prompt)
        loss = F.cross_entropy(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits, y), n=x.size(0))
    return loss_meter.avg, acc_meter.avg
