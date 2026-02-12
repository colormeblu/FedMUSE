from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running without `pip install -e .`
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from fedmuse.utils.seed import set_seed
from fedmuse.utils.io import ensure_dir, load_json, save_json, save_csv, flatten_dict
from fedmuse.data.protocols import apply_federated_protocol, build_partition_report

# Ensure registries are populated
import fedmuse.data  # noqa: F401
import fedmuse.algorithms  # noqa: F401
from fedmuse.registry import get_dataset, get_algorithm, list_algorithms, list_datasets

def _set_by_dotted_key(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _parse_value(v: str) -> Any:
    # Try JSON parsing first (handles numbers, bool, lists, dicts)
    try:
        return json.loads(v)
    except Exception:
        pass
    # fallback
    return v

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        _set_by_dotted_key(out, k.strip(), _parse_value(v.strip()))
    return out

def run_single(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    ds_cfg = cfg.get("dataset", {})
    algo_cfg = cfg.get("algorithm", {})
    model_cfg = cfg.get("model", {}) or {}
    if "name" not in ds_cfg or "name" not in algo_cfg:
        raise ValueError("Config must contain dataset.name and algorithm.name")

    dataset_name = str(ds_cfg["name"]).lower()
    algo_name = str(algo_cfg["name"]).lower()

    # Convenience: if user selects a CLIP backbone but didn't specify transforms,
    # default to CLIP-style normalization.
    backbone = str(model_cfg.get("backbone", algo_cfg.get("backbone", "resnet18"))).lower()
    if ds_cfg.get("transforms") is None and backbone in ("open_clip", "openclip", "clip"):
        ds_cfg["transforms"] = "clip"

    builder = get_dataset(dataset_name)
    algo_cls = get_algorithm(algo_name)

    base_domains = builder(
        data_root=str(ds_cfg["root"]),
        domains=ds_cfg.get("domains"),
        val_ratio=float(ds_cfg.get("val_ratio", 0.1)),
        seed=seed,
        image_size=int(ds_cfg.get("image_size", 224)),
        transforms=str(ds_cfg.get("transforms", "default")),
    )
    # Base domains are built once; federated protocol is applied per target (needed for iid/dirichlet).

    # Auto-LOO convenience: allow dataset.target_domain to be omitted or set to ALL
    # and we will run all targets sequentially in one output folder.
    target_val = ds_cfg.get("target_domain", None)
    protocol = str(ds_cfg.get("protocol", "loo")).lower()
    run_all_targets = (protocol in ("loo", "domain", "domain_loo")) and (
        target_val is None or str(target_val).lower() in ("all", "*", "none")
    )

    runner = algo_cls()

    exp = cfg.get("experiment", {})
    outdir_root = Path(exp.get("outdir", "outputs"))
    name = exp.get("name", f"{dataset_name}_{algo_name}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{name}__target={'ALL' if run_all_targets else ds_cfg['target_domain']}__seed={seed}__{stamp}"
    outdir = ensure_dir(outdir_root / run_id)

    save_json(cfg, outdir / "config.json")

    def _run_one_target(tgt: str, outdir_t: Path) -> Tuple[Dict[str, Any], float]:
        cfg_t = copy.deepcopy(cfg)
        cfg_t.setdefault("dataset", {})
        cfg_t["dataset"]["target_domain"] = tgt

        # Apply protocol per target (important for iid/dirichlet client construction).
        domains_t = apply_federated_protocol(base_domains, cfg_t["dataset"], seed=seed)

        # Persist client partition summary for reproducibility / ablations.
        part = build_partition_report(domains_t, base_domains, cfg_t["dataset"], seed=seed)
        save_json(part, outdir_t / "partition.json")

        # Initialize live outputs so files exist before training starts.
        init_history = {"round": [], "test_acc": [], "test_loss": []}
        summary_base = {
            "dataset": dataset_name,
            "target_domain": tgt,
            "algorithm": algo_name,
            "seed": seed,
            "final_acc": float("nan"),
            "final_acc_std": 0.0,
            "seconds": 0.0,
            **{f"hp.{k}": v for k, v in flatten_dict({"dataset": cfg_t.get("dataset", {}), "algorithm": cfg_t.get("algorithm", {})}).items()},
        }
        save_json(init_history, outdir_t / "history.json")
        save_json({"summary": summary_base, "raw": {"history": init_history, "final_acc": float("nan")}}, outdir_t / "result.json")
        cfg_t.setdefault("_runtime", {})
        cfg_t["_runtime"]["outdir"] = str(outdir_t)
        cfg_t["_runtime"]["summary"] = summary_base

        print(f"==> Running: dataset={dataset_name}, target={tgt}, algo={algo_name}")
        t0 = time.time()
        res = runner.run(cfg_t, domains_t)
        dt = time.time() - t0
        return res, dt

    if run_all_targets:
        targets = list(base_domains.keys())
        accs = []
        per_target = {}
        total_t = 0.0
        for tgt in targets:
            sub = ensure_dir(Path(outdir) / "targets" / f"target={tgt}")
            res, dt = _run_one_target(tgt, sub)
            total_t += dt
            if isinstance(res, dict) and "history" in res:
                save_json(res["history"], sub / "history.json")
            save_json(res, sub / "result_raw.json")
            acc = float(res.get("final_acc", np.nan))
            accs.append(acc)
            per_target[tgt] = {"final_acc": acc, "seconds": dt}
        # mean/std across targets
        vals = [a for a in accs if a == a]
        mean = float(sum(vals) / len(vals)) if vals else float("nan")
        std = float((sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5) if len(vals) > 1 else 0.0
        result = {"final_acc": mean, "final_acc_std": std, "per_target": per_target}
        dt = total_t
    else:
        result, dt = _run_one_target(ds_cfg["target_domain"], outdir)

    # save history if present
    if isinstance(result, dict) and "history" in result:
        save_json(result["history"], outdir / "history.json")

    summary = {
        "dataset": dataset_name,
        "target_domain": "ALL" if run_all_targets else ds_cfg["target_domain"],
        "algorithm": algo_name,
        "seed": seed,
        "final_acc": float(result.get("final_acc", np.nan)),
        "final_acc_std": float(result.get("final_acc_std", 0.0)),
        "seconds": dt,
        **{f"hp.{k}": v for k, v in flatten_dict({"dataset": ds_cfg, "algorithm": algo_cfg}).items()},
    }
    save_json({"summary": summary, "raw": result}, outdir / "result.json")
    return {"outdir": str(outdir), "summary": summary}

def _product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for items in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, items)})
    return combos

def run_grid(grid_cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    base_path = grid_cfg.get("base_config")
    if not base_path:
        raise ValueError("Grid config must include base_config")
    base = load_json(base_path)
    base = apply_overrides(base, overrides)

    exp_name = grid_cfg.get("experiment_name", "grid_experiment")
    outdir_root = Path(grid_cfg.get("outdir", base.get("experiment", {}).get("outdir", "outputs")))
    ensure_dir(outdir_root)

    grid = grid_cfg.get("grid", {}) or {}
    ablations = grid_cfg.get("ablations", {"full": {}}) or {"full": {}}

    combos = _product({k: v if isinstance(v, list) else [v] for k, v in grid.items()})
    rows: List[Dict[str, Any]] = []

    for abl_name, abl_over in ablations.items():
        for combo in combos:
            cfg = copy.deepcopy(base)
            # ensure experiment name
            cfg.setdefault("experiment", {})
            cfg["experiment"]["name"] = exp_name

            # apply grid values
            for k, v in combo.items():
                _set_by_dotted_key(cfg, k, v)
            # apply ablation overrides
            for k, v in (abl_over or {}).items():
                _set_by_dotted_key(cfg, k, v)

            # helpful tag
            cfg.setdefault("tags", {})
            cfg["tags"]["ablation"] = abl_name

            # run
            res = run_single(cfg)
            row = {"ablation": abl_name, **res["summary"]}
            rows.append(row)

    # write results table
    stamp = time.strftime("%Y%m%d-%H%M%S")
    table_path = outdir_root / f"{exp_name}__results_{stamp}.csv"
    save_csv(rows, table_path)

    # ablation summary (mean/std over targets & seeds)
    if rows:
        by = {}
        for r in rows:
            by.setdefault(r.get("ablation","full"), []).append(float(r.get("final_acc", float("nan"))))
        lines = ["# Ablation summary", "", "| ablation | mean_acc | std_acc | n |", "|---|---:|---:|---:|"]
        for abl, vals in sorted(by.items()):
            vals = [v for v in vals if v == v]  # drop nan
            n = len(vals)
            mean = sum(vals)/n if n else float("nan")
            std = (sum((v-mean)**2 for v in vals)/max(n-1,1))**0.5 if n>1 else 0.0
            lines.append(f"| {abl} | {mean:.4f} | {std:.4f} | {n} |")
        md_path = outdir_root / f"{exp_name}__ablation_summary_{stamp}.md"
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"results_csv": str(table_path)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Single-run config JSON file.")
    ap.add_argument("--grid", type=str, default=None, help="Grid config JSON file.")
    ap.add_argument("--set", action="append", default=[], help="Override, e.g., dataset.root=/data or algorithm.beta=0")
    ap.add_argument("--list", action="store_true", help="List available datasets and algorithms.")
    args = ap.parse_args()

    if args.list:
        print("Datasets:", ", ".join(list_datasets()))
        print("Algorithms:", ", ".join(list_algorithms()))
        return

    if (args.config is None) == (args.grid is None):
        raise SystemExit("Provide exactly one of --config or --grid")

    if args.config:
        cfg = load_json(args.config)
        cfg = apply_overrides(cfg, args.set)
        out = run_single(cfg)
        print("Saved to:", out["outdir"])
    else:
        gcfg = load_json(args.grid)
        out = run_grid(gcfg, args.set)
        print("Grid done. Results:", out["results_csv"])

if __name__ == "__main__":
    main()
