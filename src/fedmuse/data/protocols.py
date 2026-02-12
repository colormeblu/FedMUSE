from __future__ import annotations

"""Federated partition protocols.

The dataset builders in :mod:`fedmuse.data.*` return a mapping
``domain_name -> DomainData`` where each *domain* corresponds to a folder.

This module optionally converts those domain datasets into *client datasets*
according to a protocol defined in ``dataset.protocol``:

* ``loo``: Leave-One-Domain-Out. (Default) Clients are the source domains.
* ``iid``: Pool all source-domain *train* samples and randomly split into N clients.
* ``dirichlet``: Pool all source-domain train samples and split into N clients
  with non-IID label proportions sampled from a Dirichlet distribution.

The algorithms are intentionally kept unchanged: we still return a dict of
``name -> DomainData`` and keep the test set as the target domain.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Subset

from fedmuse.data.base import DomainData, get_transforms, split_indices
from fedmuse.data.domain_folder import GlobalClassImageFolder, Sample


def _extract_samples_from_dataset(ds) -> List[Sample]:
    """Extract underlying (path,label) samples from a (Subset of) GlobalClassImageFolder.

    This is used to rebuild pooled datasets with proper train/test transforms.
    """
    # Subset
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = list(ds.indices)
        if hasattr(base, "samples"):
            samples = getattr(base, "samples")
            return [samples[i] for i in idxs]
        raise TypeError("Subset dataset has no .samples attribute; can't pool samples.")

    # Plain dataset
    if hasattr(ds, "samples"):
        return list(getattr(ds, "samples"))

    raise TypeError("Dataset has no .samples attribute; can't pool samples.")


def _partition_iid(n: int, num_clients: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [s.tolist() for s in splits]


def _partition_dirichlet(labels: Sequence[int], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    """Dirichlet non-IID label distribution partition.

    Returns a list of index lists, one per client.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1 if labels.size else 0
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # For each class, split its indices across clients according to Dirichlet proportions
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if idx_c.size == 0:
            continue
        props = rng.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to counts
        counts = (props * idx_c.size).astype(int)
        # Fix rounding issues to ensure sum == size
        diff = idx_c.size - counts.sum()
        if diff > 0:
            for i in rng.choice(num_clients, size=diff, replace=True):
                counts[i] += 1
        elif diff < 0:
            # subtract extras
            for i in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
                counts[i] -= 1

        start = 0
        for k in range(num_clients):
            take = counts[k]
            if take <= 0:
                continue
            client_indices[k].extend(idx_c[start : start + take].tolist())
            start += take

    # Shuffle within each client for good measure
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    # Ensure no empty clients (best-effort): move a few samples if needed.
    empties = [k for k, ids in enumerate(client_indices) if len(ids) == 0]
    if empties:
        donors = sorted(range(num_clients), key=lambda k: len(client_indices[k]), reverse=True)
        for ek in empties:
            for dk in donors:
                if len(client_indices[dk]) > 1:
                    client_indices[ek].append(client_indices[dk].pop())
                    break

    return client_indices


def apply_federated_protocol(
    base_domains: Dict[str, DomainData],
    dataset_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, DomainData]:
    """Return a (possibly) transformed domain dict according to dataset.protocol.

    The returned dict must include the target domain under the same key as
    ``dataset_cfg['target_domain']``.
    """
    protocol = str(dataset_cfg.get("protocol", "loo")).lower()
    target = dataset_cfg.get("target_domain")

    # Default: domain-as-client protocols
    if protocol in ("loo", "domain", "domain_loo"):
        return base_domains

    if target is None:
        raise ValueError("dataset.target_domain is required for protocol=iid/dirichlet")
    if target not in base_domains:
        raise KeyError(f"Target domain '{target}' not found in dataset domains: {list(base_domains.keys())}")

    # Pool source train samples
    source_domains = dataset_cfg.get("source_domains")
    if source_domains is None:
        source_domains = [d for d in base_domains.keys() if d != target]
    else:
        source_domains = list(source_domains)
        if target in source_domains:
            source_domains = [d for d in source_domains if d != target]

    pooled_samples: List[Sample] = []
    for d in source_domains:
        pooled_samples.extend(_extract_samples_from_dataset(base_domains[d].train))

    if len(pooled_samples) == 0:
        raise RuntimeError("No pooled samples found for source domains. Check your dataset splits.")

    # Rebuild pooled datasets with train/test transforms
    image_size = int(dataset_cfg.get("image_size", 224))
    tfm_train, tfm_test = get_transforms(name=str(dataset_cfg.get("transforms", "default")), image_size=image_size)
    pooled_train_full = GlobalClassImageFolder(pooled_samples, transform=tfm_train)
    pooled_test_full = GlobalClassImageFolder(pooled_samples, transform=tfm_test)

    num_clients = int(dataset_cfg.get("num_clients", max(1, len(source_domains))))
    num_clients = max(1, num_clients)

    if protocol == "iid":
        parts = _partition_iid(len(pooled_samples), num_clients=num_clients, seed=seed)
    elif protocol in ("dirichlet", "non_iid", "noniid"):
        alpha = float(dataset_cfg.get("dirichlet_alpha", 0.3))
        labels = [s.label for s in pooled_samples]
        parts = _partition_dirichlet(labels, num_clients=num_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unknown dataset.protocol: {protocol}")

    # Create new client DomainData objects. Keep target domain unchanged.
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    class_names = base_domains[target].class_names
    num_classes = base_domains[target].num_classes

    out: Dict[str, DomainData] = {}
    for k, idxs in enumerate(parts):
        if len(idxs) == 0:
            continue
        tr_idx, va_idx = split_indices(len(idxs), val_ratio=val_ratio, seed=seed + 13 * k)
        tr = [idxs[i] for i in tr_idx.tolist()]
        va = [idxs[i] for i in va_idx.tolist()]
        out[f"client{k}"] = DomainData(
            name=f"client{k}",
            train=Subset(pooled_train_full, tr),
            val=Subset(pooled_test_full, va),
            test=Subset(pooled_test_full, va),
            num_classes=num_classes,
            class_names=list(class_names),
        )

    # Add target domain for evaluation.
    out[target] = base_domains[target]
    return out


def _infer_domain_from_path(path: str, known_domains: List[str]) -> str:
    """Best-effort inference of original domain name from an absolute/relative file path."""
    try:
        parts = list(Path(path).parts)
    except Exception:
        parts = str(path).replace('\\', '/').split('/')
    # Prefer exact segment matches.
    for seg in parts:
        if seg in known_domains:
            return seg
    # Fallback: substring match (rare)
    low = str(path).lower()
    for d in known_domains:
        if d.lower() in low:
            return d
    return "unknown"


def build_partition_report(
    domains: Dict[str, DomainData],
    base_domains: Dict[str, DomainData],
    dataset_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Summarize how samples were assigned to federated clients.

    This is intended for reproducibility and paper-quality reporting. It stores:
      - per-client sample counts (train/val)
      - per-client label histogram (index and class-name)
      - (best-effort) per-client original-domain histogram, inferred from file paths

    Note: it does *not* dump per-sample file lists to avoid huge outputs on DomainNet.
    """
    protocol = str(dataset_cfg.get("protocol", "loo")).lower()
    target = dataset_cfg.get("target_domain")
    known_domains = list(base_domains.keys())

    # Identify clients in the transformed `domains` mapping.
    client_names = [k for k in domains.keys() if k != target]

    # class meta
    first = next(iter(domains.values()))
    num_classes = int(getattr(first, "num_classes", 0))
    class_names = list(getattr(first, "class_names", [str(i) for i in range(num_classes)]))

    def _get_label(s) -> int:
        if hasattr(s, "label"):
            return int(s.label)
        if isinstance(s, (tuple, list)) and len(s) >= 2:
            return int(s[1])
        if hasattr(s, "y"):
            return int(s.y)
        if hasattr(s, "label_id"):
            return int(s.label_id)
        raise TypeError(f"Cannot extract label from sample of type {type(s)}")

    def _get_path(s) -> str:
        if hasattr(s, "path"):
            return str(s.path)
        if isinstance(s, (tuple, list)) and len(s) >= 1:
            return str(s[0])
        raise TypeError(f"Cannot extract path from sample of type {type(s)}")

    def _hist(samples) -> Dict[int, int]:
        h: Dict[int, int] = {i: 0 for i in range(num_classes)}
        for s in samples:
            yi = _get_label(s)
            h[yi] = h.get(yi, 0) + 1
        return {k: int(v) for k, v in h.items() if v != 0}

    def _domain_hist(samples) -> Dict[str, int]:
        dh: Dict[str, int] = {}
        for s in samples:
            p = _get_path(s)
            d = _infer_domain_from_path(p, known_domains)
            dh[d] = dh.get(d, 0) + 1
        return dict(sorted(dh.items(), key=lambda kv: (-kv[1], kv[0])))



    def _hist_named(h: Dict[int, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k, v in h.items():
            name = class_names[int(k)] if int(k) < len(class_names) else str(k)
            out[name] = int(v)
        return out



    clients_info: Dict[str, Any] = {}
    total_train = 0
    total_val = 0

    for c in sorted(client_names):
        tr_samples = _extract_samples_from_dataset(domains[c].train)
        va_samples = _extract_samples_from_dataset(domains[c].val)

        tr_h = _hist(tr_samples)
        va_h = _hist(va_samples)
        dh = _domain_hist(tr_samples)

        total_train += len(tr_samples)
        total_val += len(va_samples)

        clients_info[c] = {
            "n_train": int(len(tr_samples)),
            "n_val": int(len(va_samples)),
            "label_hist_train": tr_h,
            "label_hist_train_named": _hist_named(tr_h),
            "label_hist_val": va_h,
            "label_hist_val_named": _hist_named(va_h),
            "domain_hist_train": dh,
            "domain_frac_train": {k: (v / max(1, len(tr_samples))) for k, v in dh.items()},
        }

    report: Dict[str, Any] = {
        "protocol": protocol,
        "seed": int(seed),
        "target_domain": target,
        "known_domains": known_domains,
        "num_classes": num_classes,
        "class_names": class_names,
        "clients": clients_info,
        "totals": {"n_train": int(total_train), "n_val": int(total_val), "num_clients": int(len(client_names))},
    }

    # Include protocol params for convenience
    if protocol in ("iid", "dirichlet"):
        report["protocol_params"] = {
            "num_clients": int(dataset_cfg.get("num_clients", len(client_names))),
            "dirichlet_alpha": float(dataset_cfg.get("dirichlet_alpha", 0.5)),
            "val_ratio": float(dataset_cfg.get("val_ratio", 0.1)),
        }

    return report