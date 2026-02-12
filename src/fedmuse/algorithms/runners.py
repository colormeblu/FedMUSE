from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD

from fedmuse.registry import register_algorithm
from fedmuse.data.base import make_loader
from fedmuse.models.clip_backbone import build_classifier_from_cfg
from fedmuse.models.fedmuse_clip import PromptedOpenCLIPVision
from fedmuse.algorithms.agg import fedavg_state_dict, weighted_average_state_dict
from fedmuse.algorithms.local_train import train_classifier_epoch, eval_classifier
from fedmuse.algorithms.fedmuse_train import (
    StyleMappingNetwork,
    collect_feature_stats,
    eval_fedmuse_ua_ttaf,
    train_fedmuse_epoch,
)
from fedmuse.utils.io import save_json


def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _build_clients(domains: Dict[str, Any], target_domain: str, batch_size: int, num_workers: int):
    clients = []
    for name, dd in domains.items():
        if name == target_domain:
            continue
        clients.append(
            {
                "name": name,
                "train_loader": make_loader(dd.train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                "val_loader": make_loader(dd.val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
                "train_dataset": dd.train,
                "n_train": len(dd.train),
            }
        )
    test_loader = make_loader(domains[target_domain].test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return clients, test_loader


def _save_history(outdir: Path, history: Dict[str, Any]):
    save_json(history, outdir / "history.json")


def _update_live_results(cfg: Dict[str, Any], history: Dict[str, Any], final_acc: float, seconds: float) -> None:
    runtime = cfg.get("_runtime") if isinstance(cfg, dict) else None
    if not runtime:
        return
    outdir = runtime.get("outdir")
    if not outdir:
        return
    outdir = Path(outdir)
    _save_history(outdir, history)

    summary = runtime.get("summary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    summary["final_acc"] = float(final_acc)
    summary["seconds"] = float(seconds)
    save_json({"summary": summary, "raw": {"history": history, "final_acc": float(final_acc)}}, outdir / "result.json")


def _norm_weights_from_counts(counts: List[int]) -> List[float]:
    total = float(sum(int(c) for c in counts))
    if total <= 0:
        return [1.0 / max(1, len(counts)) for _ in counts]
    return [float(c) / total for c in counts]


def _format_class_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ")


def _template_text(template: str, class_name: str) -> str:
    cname = _format_class_name(class_name)
    if "{class}" in template:
        return template.format(**{"class": cname})
    if "{}" in template:
        return template.format(cname)
    return f"{template} {cname}"


def _style_text(template: str, domain_name: str) -> str:
    dname = _format_class_name(domain_name)
    if "{domain}" in template:
        return template.format(**{"domain": dname})
    if "{}" in template:
        return template.format(dname)
    return f"{template} {dname}"


def _build_style_texts(client_names: Sequence[str], templates: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tpl in templates:
        tpl = str(tpl)
        for cname in client_names:
            t = _style_text(tpl, cname)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _build_prompt_optimizer(
    model: PromptedOpenCLIPVision,
    style_mapper: StyleMappingNetwork,
    algo: Dict[str, Any],
) -> torch.optim.Optimizer:
    opt_name = str(algo.get("optimizer", "adamw")).lower().strip()
    lr = float(algo.get("lr", 1e-4))
    wd = float(algo.get("wd", 1e-4))
    params = list(model.prompt.parameters()) + list(style_mapper.parameters())
    if opt_name == "sgd":
        momentum = float(algo.get("momentum", 0.9))
        return SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _mean_metric(metrics: List[Dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return float(sum(float(m.get(key, 0.0)) for m in metrics) / float(len(metrics)))


class ExpertRouter(nn.Module):
    """Feature->expert logits router used by UA-TTAF."""

    def __init__(self, feature_dim: int, num_experts: int, hidden_dim: int = 256) -> None:
        super().__init__()
        h = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), h),
            nn.GELU(),
            nn.Linear(h, int(num_experts)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_router_on_centers(
    router: Optional[ExpertRouter],
    centers: Sequence[torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
) -> float:
    if router is None or len(centers) <= 1 or int(steps) <= 0:
        return 0.0
    x = torch.stack([c.to(device=device, dtype=torch.float32) for c in centers], dim=0)
    y = torch.arange(len(centers), device=device, dtype=torch.long)

    opt = torch.optim.Adam(router.parameters(), lr=max(1e-6, float(lr)))
    router.train()
    loss_val = 0.0
    for _ in range(int(steps)):
        logits = router(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_val = float(loss.item())
    router.eval()
    return loss_val


@register_algorithm("fedavg")
class FedAvgRunner:
    """Simple supervised FedAvg baseline (classifier finetuning)."""

    def run(self, cfg: Dict[str, Any], domains: Dict[str, Any]) -> Dict[str, Any]:
        algo = cfg["algorithm"]
        target = cfg["dataset"]["target_domain"]

        rounds = int(algo.get("rounds", 100))
        local_epochs = int(algo.get("local_epochs", 1))
        batch_size = int(algo.get("batch_size", 32))
        lr = float(algo.get("lr", 0.001))
        momentum = float(algo.get("momentum", 0.9))
        wd = float(algo.get("wd", 5e-4))
        show_progress = bool(algo.get("tqdm", True))

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)
        num_classes = next(iter(domains.values())).num_classes

        clients, test_loader = _build_clients(domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")
        global_model, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device, pretrained=True)

        history = {"round": [], "test_acc": [], "test_loss": []}
        t_start = time.time()

        for r in range(1, rounds + 1):
            updates = []
            for c in clients:
                local, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device)
                local.load_state_dict(global_model.state_dict(), strict=True)
                opt = SGD(local.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

                for e in range(local_epochs):
                    progress = None
                    if show_progress:
                        progress = {
                            "enabled": True,
                            "desc": f"round {r} client {c['name']} epoch {e + 1}/{local_epochs}",
                            "leave": False,
                            "dynamic_ncols": True,
                        }
                    train_classifier_epoch(local, c["train_loader"], opt, device, progress=progress)

                updates.append((local.state_dict(), int(c["n_train"])))

            avg_sd = fedavg_state_dict(updates)
            global_model.load_state_dict(avg_sd, strict=True)

            test_loss, test_acc = eval_classifier(global_model, test_loader, device)
            history["round"].append(r)
            history["test_acc"].append(float(test_acc))
            history["test_loss"].append(float(test_loss))

            _update_live_results(cfg, history, final_acc=float(test_acc), seconds=time.time() - t_start)
            if r == 1 or r % int(algo.get("log_every", 1)) == 0:
                print(f"[Round {r:03d}] test_acc={test_acc * 100:.2f} test_loss={test_loss:.4f}")

        return {"history": history, "final_acc": float(history["test_acc"][-1])}


@register_algorithm("fedmuse")
class FedMUSERunner:
    """Fed-MUSE runner with SGDH + HD-SPT + UA-TTAF."""

    def run(self, cfg: Dict[str, Any], domains: Dict[str, Any]) -> Dict[str, Any]:
        algo = cfg["algorithm"]
        ds_cfg = cfg["dataset"]
        model_cfg = cfg.get("model", {}) or {}
        target = ds_cfg["target_domain"]

        rounds = int(algo.get("rounds", 50))
        local_epochs = int(algo.get("local_epochs", 1))
        batch_size = int(algo.get("batch_size", 32))
        lambda_sem = float(algo.get("lambda_sem", 0.5))
        lambda_orth = float(algo.get("lambda_orth", 0.1))
        use_semantic_hallucination = bool(algo.get("use_semantic_hallucination", True))
        use_orth_loss = bool(algo.get("use_orth_loss", True))
        use_ua_ttaf = bool(algo.get("use_ua_ttaf", True))
        beta_uncertainty = float(algo.get("beta_uncertainty", 1.0))
        gamma_uncertainty = float(algo.get("gamma_uncertainty", 0.0))
        prompt_len = int(algo.get("prompt_len", 4))
        prompt_layers = int(algo.get("prompt_layers", 4))
        prompt_start_layer = algo.get("prompt_start_layer")
        if prompt_start_layer is not None:
            prompt_start_layer = int(prompt_start_layer)
        style_mapper_hidden = int(algo.get("style_mapper_hidden", 1024))
        router_hidden = int(algo.get("router_hidden", 256))
        router_steps = int(algo.get("router_steps", 100))
        router_lr = float(algo.get("router_lr", 1e-3))
        stat_max_batches = int(algo.get("stat_max_batches", -1))
        show_progress = bool(algo.get("tqdm", True))

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)

        clients, test_loader = _build_clients(domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")

        open_clip_cfg = (model_cfg.get("open_clip", {}) or {})
        model_name = str(open_clip_cfg.get("name", "ViT-B-16"))
        pretrained = str(open_clip_cfg.get("pretrained", "openai"))

        model = PromptedOpenCLIPVision(
            model_name=model_name,
            pretrained=pretrained,
            prompt_len=prompt_len,
            prompt_layers=prompt_layers,
            prompt_start_layer=prompt_start_layer,
        ).to(device)
        model.freeze_backbone()

        class_names = list(domains[target].class_names)
        class_template = str(algo.get("class_template", "a photo of a {}"))
        class_texts = [_template_text(class_template, c) for c in class_names]
        class_text_features = model.encode_text(class_texts, device=device, normalize=True)
        feature_dim = int(class_text_features.size(1))

        style_mapper = StyleMappingNetwork(
            text_dim=feature_dim,
            feature_dim=feature_dim,
            hidden_dim=style_mapper_hidden,
        ).to(device)

        source_names = [str(c["name"]) for c in clients]
        style_templates = algo.get(
            "style_templates",
            [
                "a photo in the style of {domain}",
                "an image from {domain} domain",
            ],
        )
        if not isinstance(style_templates, list) or len(style_templates) == 0:
            style_templates = ["a photo in the style of {domain}"]
        style_texts = _build_style_texts(source_names, [str(t) for t in style_templates])
        if bool(use_semantic_hallucination) and len(style_texts) > 0:
            style_text_features = model.encode_text(style_texts, device=device, normalize=True)
        else:
            style_text_features = None

        logit_scale = float(algo.get("logit_scale", model.get_logit_scale()))

        global_prompt_state = model.global_prompt_state_dict_cpu()
        global_style_state = style_mapper.state_dict_cpu()
        init_local_state = model.local_prompt_state_dict_cpu()
        client_local_states: Dict[str, Dict[str, torch.Tensor]] = {
            c["name"]: {k: v.clone() for k, v in init_local_state.items()} for c in clients
        }

        router: Optional[ExpertRouter]
        if len(clients) > 1:
            router = ExpertRouter(feature_dim=feature_dim, num_experts=len(clients), hidden_dim=router_hidden).to(device)
        else:
            router = None

        history = {
            "round": [],
            "test_acc": [],
            "test_loss": [],
            "train_ce_loss": [],
            "train_sem_loss": [],
            "train_orth_loss": [],
            "train_total_loss": [],
            "agg_weights": [],
            "mean_alpha": [],
            "router_loss": [],
            "client_names": source_names,
        }
        t_start = time.time()

        for r in range(1, rounds + 1):
            global_updates: List[Dict[str, torch.Tensor]] = []
            style_updates: List[Dict[str, torch.Tensor]] = []
            client_sizes: List[int] = []
            client_metrics: List[Dict[str, float]] = []

            expert_prompts: List[torch.Tensor] = []
            expert_means: List[torch.Tensor] = []
            expert_vars: List[torch.Tensor] = []

            for c in clients:
                cname = str(c["name"])
                model.load_global_prompt_state_dict(global_prompt_state)
                model.load_local_prompt_state_dict(client_local_states[cname])
                style_mapper.load_state_dict(global_style_state, strict=True)

                opt = _build_prompt_optimizer(model, style_mapper, algo)
                epoch_metrics: List[Dict[str, float]] = []
                for e in range(local_epochs):
                    progress = None
                    if show_progress:
                        progress = {
                            "enabled": True,
                            "desc": f"round {r} client {cname} epoch {e + 1}/{local_epochs}",
                            "leave": False,
                            "dynamic_ncols": True,
                        }
                    m = train_fedmuse_epoch(
                        model=model,
                        style_mapper=style_mapper,
                        loader=c["train_loader"],
                        class_text_features=class_text_features,
                        style_text_features=style_text_features,
                        optimizer=opt,
                        device=device,
                        lambda_sem=lambda_sem,
                        lambda_orth=lambda_orth,
                        logit_scale=logit_scale,
                        use_semantic_hallucination=use_semantic_hallucination,
                        use_orth_loss=use_orth_loss,
                        progress=progress,
                    )
                    epoch_metrics.append(m)

                global_updates.append(model.global_prompt_state_dict_cpu())
                style_updates.append(_state_dict_cpu(style_mapper))
                local_state = model.local_prompt_state_dict_cpu()
                client_local_states[cname] = local_state
                client_sizes.append(int(c["n_train"]))

                mu, var = collect_feature_stats(
                    model=model,
                    loader=c["train_loader"],
                    device=device,
                    prompt_mode="joint",
                    max_batches=stat_max_batches,
                )
                expert_prompts.append(PromptedOpenCLIPVision.effective_prompt_from_state(local_state, stream="local"))
                expert_means.append(mu)
                expert_vars.append(var)

                client_metrics.append(
                    {
                        "ce_loss": _mean_metric(epoch_metrics, "ce_loss"),
                        "sem_loss": _mean_metric(epoch_metrics, "sem_loss"),
                        "orth_loss": _mean_metric(epoch_metrics, "orth_loss"),
                        "total_loss": _mean_metric(epoch_metrics, "total_loss"),
                    }
                )

            weights = _norm_weights_from_counts(client_sizes)
            global_prompt_state = weighted_average_state_dict(global_updates, weights)
            global_style_state = weighted_average_state_dict(style_updates, weights)

            router_loss = _fit_router_on_centers(
                router=router,
                centers=expert_means,
                device=device,
                steps=router_steps,
                lr=router_lr,
            )

            model.load_global_prompt_state_dict(global_prompt_state)
            style_mapper.load_state_dict(global_style_state, strict=True)
            global_prompt = PromptedOpenCLIPVision.effective_prompt_from_state(global_prompt_state, stream="global")

            test_loss, test_acc, mean_alpha = eval_fedmuse_ua_ttaf(
                model=model,
                loader=test_loader,
                class_text_features=class_text_features,
                device=device,
                logit_scale=logit_scale,
                global_prompt=global_prompt,
                expert_prompts=expert_prompts,
                expert_means=expert_means,
                expert_vars=expert_vars,
                beta=beta_uncertainty,
                gamma=gamma_uncertainty,
                router=router,
                use_ua_ttaf=use_ua_ttaf,
            )

            round_ce = float(sum(weights[i] * client_metrics[i]["ce_loss"] for i in range(len(client_metrics))))
            round_sem = float(sum(weights[i] * client_metrics[i]["sem_loss"] for i in range(len(client_metrics))))
            round_orth = float(sum(weights[i] * client_metrics[i]["orth_loss"] for i in range(len(client_metrics))))
            round_total = float(sum(weights[i] * client_metrics[i]["total_loss"] for i in range(len(client_metrics))))

            history["round"].append(r)
            history["test_acc"].append(float(test_acc))
            history["test_loss"].append(float(test_loss))
            history["train_ce_loss"].append(round_ce)
            history["train_sem_loss"].append(round_sem)
            history["train_orth_loss"].append(round_orth)
            history["train_total_loss"].append(round_total)
            history["agg_weights"].append([float(w) for w in weights])
            history["mean_alpha"].append(float(mean_alpha))
            history["router_loss"].append(float(router_loss))

            _update_live_results(cfg, history, final_acc=float(test_acc), seconds=time.time() - t_start)

            if r == 1 or r % int(algo.get("log_every", 1)) == 0:
                print(
                    f"[Round {r:03d}] test_acc={test_acc * 100:.2f} test_loss={test_loss:.4f} "
                    f"total={round_total:.4f} ce={round_ce:.4f} sem={round_sem:.4f} "
                    f"orth={round_orth:.4f} alpha={mean_alpha:.3f}"
                )

        return {"history": history, "final_acc": float(history["test_acc"][-1])}
