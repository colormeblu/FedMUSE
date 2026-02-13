from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageOps

@dataclass
class DomainData:
    name: str
    train: Dataset
    val: Dataset
    test: Dataset
    num_classes: int
    class_names: List[str]

def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val)) if n > 1 else 0
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx

# ---- Minimal transforms (no torchvision dependency) ----

class Compose:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

class Resize:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size, self.size), resample=Image.BILINEAR)

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            return ImageOps.mirror(img)
        return img

class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,C]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]
        return t

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

def default_transforms(image_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm_train = Compose([
        Resize(image_size),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean, std),
    ])
    tfm_test = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean, std),
    ])
    return tfm_train, tfm_test


def clip_transforms(image_size: int = 224):
    """CLIP-style normalization (OpenAI / OpenCLIP).

    Uses the common CLIP mean/std so you can train/eval with a CLIP backbone
    without relying on torchvision's preprocessing pipeline.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    tfm_train = Compose([
        Resize(image_size),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean, std),
    ])
    tfm_test = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean, std),
    ])
    return tfm_train, tfm_test


def get_transforms(name: str = "default", image_size: int = 224):
    name = (name or "default").lower()
    if name in ("clip", "open_clip", "openclip"):
        return clip_transforms(image_size=image_size)
    return default_transforms(image_size=image_size)

def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    sampler = None
    if distributed and int(world_size) > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=bool(drop_last),
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(drop_last),
    )
