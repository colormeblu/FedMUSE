from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Subset

from fedmuse.registry import register_dataset
from fedmuse.data.base import DomainData, split_indices, get_transforms
from fedmuse.data.domain_folder import GlobalClassImageFolder, scan_domains_for_classes, build_samples

DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

@register_dataset("domainnet")
def build_domainnet_domains(
    data_root: str,
    domains: Optional[List[str]] = None,
    val_ratio: float = 0.1,
    seed: int = 0,
    image_size: int = 224,
    transforms: str = "default",
) -> Dict[str, DomainData]:
    """DomainNet: expects <data_root>/DomainNet/<domain>/<class>/*"""
    root = Path(data_root) / "DomainNet"
    domains = domains or DOMAINNET_DOMAINS

    tfm_train, tfm_test = get_transforms(name=transforms, image_size=image_size)

    domain_dirs = [root / d for d in domains]
    class_names = scan_domains_for_classes(domain_dirs)
    num_classes = len(class_names)
    if num_classes == 0:
        raise RuntimeError(f"No classes found under {root}. Check your DomainNet folder structure.")

    out: Dict[str, DomainData] = {}
    for d in domains:
        ddir = root / d
        samples = build_samples(ddir, class_names)
        n = len(samples)
        tr_idx, va_idx = split_indices(n, val_ratio=val_ratio, seed=seed)
        ds_train_full = GlobalClassImageFolder(samples, transform=tfm_train)
        ds_val_full = GlobalClassImageFolder(samples, transform=tfm_test)
        ds_test = GlobalClassImageFolder(samples, transform=tfm_test)
        train = Subset(ds_train_full, tr_idx.tolist())
        val = Subset(ds_val_full, va_idx.tolist())
        out[d] = DomainData(
            name=d,
            train=train,
            val=val,
            test=ds_test,
            num_classes=num_classes,
            class_names=list(class_names),
        )
    return out
