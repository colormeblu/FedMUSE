from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedmuse.utils.metrics import accuracy_top1, AverageMeter

@dataclass
class TrainResult:
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    train_loss: float
    train_acc: float

def _prox_penalty(model: nn.Module, global_state: Dict[str, torch.Tensor], mu: float, device: torch.device) -> torch.Tensor:
    if mu <= 0:
        return torch.tensor(0.0, device=device)
    pen = torch.tensor(0.0, device=device)
    for (name, p) in model.named_parameters():
        if not p.requires_grad:
            continue
        g = global_state[name].to(device)
        pen = pen + torch.sum((p - g) ** 2)
    return 0.5 * float(mu) * pen

def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    geo_ctx: Optional[dict] = None,
    prox_ctx: Optional[dict] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    model.train()
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

        logits, feat, feat_map = model(x)
        ce = F.cross_entropy(logits, y)

        # Optional geometry/prototype alignment regularizer (lightweight)
        loss = ce
        if geo_ctx is not None and float(geo_ctx.get("beta", 0.0)) > 0:
            mu_g = geo_ctx.get("mu_global", None)
            if mu_g is not None:
                beta = float(geo_ctx["beta"])
                mu_global = mu_g.to(device)  # [C,D]
                present = torch.unique(y)
                geo_loss = torch.tensor(0.0, device=device)
                if len(present) >= 2:
                    batch_proto = []
                    global_proto = []
                    for c in present.tolist():
                        if c < 0 or c >= mu_global.size(0):
                            continue
                        m = (y == c)
                        if int(m.sum()) == 0:
                            continue
                        batch_proto.append(feat[m].mean(dim=0))
                        global_proto.append(mu_global[c])
                    if len(batch_proto) >= 2:
                        batch_proto = torch.stack(batch_proto, dim=0)
                        global_proto = torch.stack(global_proto, dim=0)
                        Bd = torch.cdist(batch_proto, batch_proto, p=2)
                        Gd = torch.cdist(global_proto, global_proto, p=2)
                        geo_loss = F.mse_loss(Bd, Gd) + F.mse_loss(batch_proto, global_proto)
                loss = ce + beta * geo_loss

        # FedProx proximal penalty
        if prox_ctx is not None and prox_ctx.get("mu", 0.0) > 0:
            loss = loss + _prox_penalty(model, prox_ctx["global_state"], float(prox_ctx["mu"]), device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits.detach(), y), n=x.size(0))

    if pbar is not None:
        pbar.close()

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def eval_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, feat, feat_map = model(x)
        loss = F.cross_entropy(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits, y), n=x.size(0))
    return loss_meter.avg, acc_meter.avg
