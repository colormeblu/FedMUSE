from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, List

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from fedmuse.data.base import make_loader, get_transforms
from fedmuse.data.domain_folder import IMG_EXTS, pil_loader

class FlatImageDataset(Dataset):
    """Recursively scans a directory for images and returns (image, 0)."""
    def __init__(self, root: str, transform=None, max_images: Optional[int] = None):
        self.root = Path(root)
        self.transform = transform
        paths: List[str] = []
        for r, _, files in os.walk(self.root):
            for fn in files:
                if Path(fn).suffix.lower() in IMG_EXTS:
                    paths.append(str(Path(r) / fn))
        paths.sort()
        if max_images is not None:
            paths = paths[:max_images]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = pil_loader(self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

def build_public_loader_from_clients(
    client_train_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    num_public: int,
    seed: int,
    image_size: int = 224,
    transforms: str = "default",
) -> DataLoader:
    """Creates a small 'public' unlabeled loader by sampling from client train sets.

    NOTE: This is a *scaffold* default (not privacy-preserving). For a truly public set,
    use `build_public_loader_from_dir` with external images.
    """
    rng = np.random.RandomState(seed)
    concat = ConcatDataset(list(client_train_datasets))
    n = len(concat)
    if n == 0:
        raise RuntimeError("No samples available to build a public set.")
    k = min(int(num_public), n)
    idx = rng.choice(n, size=k, replace=False).tolist()
    subset = Subset(concat, idx)

    _, tfm_test = get_transforms(name=transforms, image_size=image_size)

    class _Wrap(Dataset):
        def __init__(self, ds, tfm):
            self.ds = ds
            self.tfm = tfm
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            x, _ = self.ds[i]
            # x might already be a tensor if upstream applied transforms
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x, 0
            except Exception:
                pass
            return self.tfm(x), 0

    subset = _Wrap(subset, tfm_test)
    return make_loader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def build_public_loader_from_dir(
    public_root: str,
    batch_size: int,
    num_workers: int,
    num_public: int,
    image_size: int = 224,
    transforms: str = "default",
) -> DataLoader:
    _, tfm_test = get_transforms(name=transforms, image_size=image_size)
    ds = FlatImageDataset(public_root, transform=tfm_test, max_images=int(num_public))
    return make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
