from __future__ import annotations

from typing import Any, Callable, Dict, Type

DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {}
ALGORITHM_REGISTRY: Dict[str, Type] = {}

def register_dataset(name: str):
    name_l = name.lower()
    def deco(fn: Callable[..., Any]):
        if name_l in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{name_l}' already registered")
        DATASET_REGISTRY[name_l] = fn
        return fn
    return deco

def register_algorithm(name: str):
    name_l = name.lower()
    def deco(cls: Type):
        if name_l in ALGORITHM_REGISTRY:
            raise KeyError(f"Algorithm '{name_l}' already registered")
        ALGORITHM_REGISTRY[name_l] = cls
        return cls
    return deco

def get_dataset(name: str):
    return DATASET_REGISTRY[name.lower()]

def get_algorithm(name: str):
    return ALGORITHM_REGISTRY[name.lower()]

def list_datasets():
    return sorted(DATASET_REGISTRY.keys())

def list_algorithms():
    return sorted(ALGORITHM_REGISTRY.keys())
