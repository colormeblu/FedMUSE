from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

@dataclass
class Sample:
    path: str
    label: int

class GlobalClassImageFolder(Dataset):
    """ImageFolder-like dataset with a *global* class_to_idx mapping shared across domains.

    This avoids the classic pitfall where each domain has its own class index ordering,
    which breaks federated training when you aggregate models across domains.
    """

    def __init__(self, samples: Sequence[Sample], transform: Optional[Callable] = None, loader: Callable = pil_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self.loader(s.path)
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label

def scan_domains_for_classes(domain_dirs: Sequence[Path]) -> List[str]:
    classes = set()
    for d in domain_dirs:
        if not d.exists():
            continue
        for name in os.listdir(d):
            p = d / name
            if p.is_dir():
                classes.add(name)
    return sorted(classes)

def build_samples(domain_dir: Path, class_names: Sequence[str]) -> List[Sample]:
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples: List[Sample] = []
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain dir not found: {domain_dir}")
    for cls in os.listdir(domain_dir):
        cls_dir = domain_dir / cls
        if not cls_dir.is_dir():
            continue
        if cls not in class_to_idx:
            continue
        label = class_to_idx[cls]
        for root, _, files in os.walk(cls_dir):
            for fn in files:
                ext = Path(fn).suffix.lower()
                if ext in IMG_EXTS:
                    samples.append(Sample(path=str(Path(root)/fn), label=label))
    return samples
