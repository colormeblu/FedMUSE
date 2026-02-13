# Fed-MUSE

`Fed-MUSE`(Federated Multi-modal Uncertainty-aware Style Experts) is a runnable Python project for the algorithm in `/home/dengyu/Fed-MUSE.docx`, with the same scaffold style as `FedSVP`:

- `train.py`
- `configs/*.json`
- `src/fedmuse/...`
- `tests/...`

Implemented modules:

- `SGDH` (Semantic-Guided Domain Hallucination): text-style embeddings -> style statistics -> AdaIN feature hallucination + KL semantic consistency.
- `HD-SPT` (Hierarchical Dual-Stream Prompt Tuning): shared/global prompts and local/expert prompts with orthogonal disentanglement.
- `UA-TTAF` (Uncertainty-Aware Test-Time Adaptive Fusion): Mahalanobis expert matching + router prior + entropy gate for dynamic prompt fusion.

## Install

Use Python 3.9 as specified in `pyproject.toml` and `requirements.txt`.

```bash
cd /home/dengyu/Fed-MUSE
pip install -r requirements.txt
pip install -e .
```

## Quick Start

List available datasets and algorithms:

```bash
python train.py --list
```

Single run:

```bash
python train.py --config configs/pacs_fedmuse.json \
  --set dataset.root=/path/to/data \
  --set dataset.target_domain=sketch
```

LODO over all targets:

```bash
python train.py --config configs/pacs_fedmuse.json \
  --set dataset.root=/path/to/data \
  --set dataset.target_domain=ALL
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=2 train.py --config configs/domainnet_fedmuse.json \
  --set dataset.root=/path/to/data \
  --set parallel_mode=ddp \
  --set multi_gpu=true \
  --set gpu_ids=[0,1]
```

Notes:
- DDP requires `torchrun`; do not use plain `python train.py` for multi-GPU.
- `multi_gpu=true` with empty `gpu_ids` uses all visible CUDA devices.
- You can set explicit devices, e.g. `gpu_ids=[1,2,3]`, and then use `--nproc_per_node=3`.
- Keep `device` unset (or set it to `cuda`) unless you need a specific single-GPU run.

## Module Mapping

- Model and prompt definition:
  - `src/fedmuse/models/fedmuse_clip.py`
- Training losses and SGDH/UA-TTAF core functions:
  - `src/fedmuse/algorithms/fedmuse_train.py`
- Federated runner (client training + server aggregation + test-time fusion):
  - `src/fedmuse/algorithms/runners.py`

## Configs

- Fed-MUSE:
  - `configs/pacs_fedmuse.json`
  - `configs/office_fedmuse.json`
  - `configs/domainnet_fedmuse.json`
- Baseline FedAvg:
  - `configs/pacs_fedavg.json`
  - `configs/office_fedavg.json`
  - `configs/domainnet_fedavg.json`
- Ablation grid:
  - `configs/grid_pacs_loo_ablation_fedmuse.json`

Run ablation:

```bash
python train.py --grid configs/grid_pacs_loo_ablation_fedmuse.json \
  --set dataset.root=/path/to/data
```

## Dataset Layout

PACS:

```text
<DATA_ROOT>/PACS/
  photo/<class>/*.jpg
  art/<class>/*.jpg
  cartoon/<class>/*.jpg
  sketch/<class>/*.jpg
```

Office-Home:

```text
<DATA_ROOT>/OfficeHome/
  Art/<class>/*.jpg
  Clipart/<class>/*.jpg
  Product/<class>/*.jpg
  RealWorld/<class>/*.jpg
```

DomainNet:

```text
<DATA_ROOT>/DomainNet/
  clipart/<class>/*
  infograph/<class>/*
  painting/<class>/*
  quickdraw/<class>/*
  real/<class>/*
  sketch/<class>/*
```
