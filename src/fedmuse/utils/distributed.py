from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def _parse_gpu_ids(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        s = raw.strip()
        if len(s) == 0:
            return []
        if s.startswith("[") and s.endswith("]"):
            import json

            vals = json.loads(s)
            if not isinstance(vals, list):
                raise ValueError("gpu_ids string JSON must decode to list")
            return [int(x) for x in vals]
        return [int(x.strip()) for x in s.split(",") if len(x.strip()) > 0]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise ValueError(f"gpu_ids must be list/int/string, got {type(raw).__name__}")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return int(dist.get_rank())


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_distributed():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=int(src))
    return payload[0]


def setup_distributed_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize DDP from torchrun environment when requested in config.

    Expected launch:
      torchrun --nproc_per_node=<N> train.py --config ...
    """

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    rank_env = int(os.environ.get("RANK", "0"))

    parallel_mode = str(cfg.get("parallel_mode", "ddp")).lower()
    want_ddp = bool(cfg.get("multi_gpu", False)) and parallel_mode == "ddp"

    if world_size_env > 1 and not want_ddp:
        raise RuntimeError(
            "Detected torchrun distributed launch (WORLD_SIZE>1), but config does not enable "
            "DDP (`multi_gpu=true`, optionally `parallel_mode=ddp`)."
        )

    if not want_ddp:
        return {
            "requested": False,
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device_id": None,
        }

    if world_size_env <= 1:
        return {
            "requested": True,
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device_id": None,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but CUDA is not available.")

    gpu_ids = _parse_gpu_ids(cfg.get("gpu_ids", None))
    if len(gpu_ids) > 0:
        if local_rank_env >= len(gpu_ids):
            raise ValueError(
                f"LOCAL_RANK={local_rank_env} is out of range for gpu_ids={gpu_ids}. "
                f"Launch nproc_per_node must be <= len(gpu_ids)."
            )
        device_id = int(gpu_ids[local_rank_env])
    else:
        device_id = int(local_rank_env)

    torch.cuda.set_device(device_id)

    if not is_distributed():
        backend = str(cfg.get("distributed_backend", "nccl"))
        dist.init_process_group(backend=backend, init_method="env://")

    return {
        "requested": True,
        "enabled": True,
        "rank": int(rank_env if not is_distributed() else dist.get_rank()),
        "world_size": int(world_size_env if not is_distributed() else dist.get_world_size()),
        "local_rank": int(local_rank_env),
        "device_id": int(device_id),
    }


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
