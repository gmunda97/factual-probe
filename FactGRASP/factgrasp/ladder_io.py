# read a model's saved ladder json and its per seed stage 2 rows

import json
import os

from factgrasp import paths


def _ladder(slug: str, suffix: str = "") -> dict:
    p = os.path.join(paths.RUNS, f"ladder_{slug}{suffix}.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}


def _per_seed(blob: dict, layer=None) -> dict:
    s = blob.get("summary") or {}
    best = layer if layer is not None else s.get("best_layer")
    out = {}
    for k, v in (blob.get("rows") or {}).items():
        if not k.startswith("{") or not isinstance(v, dict) or "MRR" not in v:
            continue
        try:
            cfg = json.loads(k)
        except Exception:
            continue
        if cfg.get("stage") != 2 or cfg.get("layer") != best or "seed" not in cfg:
            continue
        out.setdefault(cfg.get("row"), {})[int(cfg["seed"])] = float(v["MRR"])
    return out
