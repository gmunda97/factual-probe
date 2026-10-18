# every experiment writes one json under results/runs with its config and rows
# a rerun skips rows that are already done so a killed job just resumes

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional, Sequence

from factgrasp import paths

VERDICTS = ("green", "amber", "red", "n/a")


def config_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if _keepable(v)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj if _keepable(v)]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                v = fn()
            except Exception:
                continue
            if attr == "item":
                return v
            if isinstance(v, list) and len(v) <= 64:
                return v
            return None
    return str(obj)


def _keepable(v: Any) -> bool:
    shape = getattr(v, "shape", None)
    if shape is not None and getattr(v, "ndim", 0) >= 1:
        try:
            return int(v.numel()) <= 64
        except Exception:
            return False
    return True


class Run:

    def __init__(self, name: str, tag: str, config: Optional[dict] = None,
                 thresholds: Optional[dict] = None, out_dir: Optional[str] = None):
        self.name = name
        self.tag = tag
        self.dir = out_dir or paths.RUNS
        os.makedirs(self.dir, exist_ok=True)
        self.out_path = os.path.join(self.dir, "%s_%s.json" % (name, tag))
        self.data = {"name": name, "tag": tag, "config": {}, "rows": {},
                     "summary": {}, "thresholds": {}, "verdict": "n/a", "notes": []}
        if os.path.exists(self.out_path):
            try:
                with open(self.out_path) as fh:
                    old = json.load(fh)
                if isinstance(old, dict):
                    self._resumed_config = dict(old.get("config") or {})
                    self.data.update({k: old.get(k, self.data[k]) for k in self.data})
            except (json.JSONDecodeError, OSError):
                self.note("previous %s was unreadable and was discarded" % self.out_path)
        if config:
            self.set_config(config)
        if thresholds:
            self.set_thresholds(thresholds)

    def done(self, key: str) -> bool:
        return key in self.data["rows"]

    def get(self, key: str, default=None):
        return self.data["rows"].get(key, default)

    def put(self, key: str, value: dict) -> None:
        self.data["rows"][str(key)] = _jsonable(value)
        self.flush()

    def drop(self, key: str) -> None:
        self.data["rows"].pop(str(key), None)
        self.flush()

    def rows(self) -> dict:
        return self.data["rows"]

    IDENTITY = ("model", "encoder", "dataset", "pooling", "loss", "pool_cap", "frame")

    def set_config(self, cfg: dict) -> None:
        prev = getattr(self, "_resumed_config", None)
        if prev:
            new = _jsonable(cfg)
            clash = [(k, prev[k], new[k]) for k in self.IDENTITY
                     if prev.get(k) and new.get(k) and prev[k] != new[k]]
            if clash:
                raise RuntimeError(
                    "registry: %s already holds a run with %s, but this run has %s. Two "
                    "experiments cannot share a tag; pass a distinct --tag." % (
                        self.out_path,
                        ", ".join("%s=%r" % (k, a) for k, a, _ in clash),
                        ", ".join("%s=%r" % (k, b) for k, _, b in clash)))
        merged = _jsonable(cfg)
        for k in self.IDENTITY:
            if not merged.get(k) and self.data["config"].get(k):
                merged.pop(k, None)
        self.data["config"].update(merged)
        self.flush()

    def set_thresholds(self, th: dict) -> None:
        self.data["thresholds"].update(_jsonable(th))
        self.flush()

    MERGE_SUBDICTS = ("measured", "verdicts", "detail")

    def set_summary(self, s: dict) -> None:
        incoming = _jsonable(s)
        for key in self.MERGE_SUBDICTS:
            old_sub = self.data["summary"].get(key)
            new_sub = incoming.get(key)
            if isinstance(old_sub, dict) and isinstance(new_sub, dict):
                merged = dict(old_sub)
                merged.update(new_sub)
                incoming[key] = merged
        self.data["summary"].update(incoming)
        self.flush()

    def set_verdict(self, v: str) -> None:
        if v not in VERDICTS:
            raise ValueError("verdict must be one of %s" % (VERDICTS,))
        self.data["verdict"] = v
        self.flush()

    def note(self, text: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data.setdefault("notes", []).append("[%s] %s" % (stamp, text))
        self.flush()

    def aux_path(self, suffix: str) -> str:
        return os.path.join(self.dir, "%s_%s_%s" % (self.name, self.tag, suffix))

    def flush(self) -> None:
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=False)
        os.replace(tmp, self.out_path)

    def summary(self, rows: Sequence) -> str:
        rows = list(rows)
        head = "=== SUMMARY %s %s ===" % (self.name, self.tag)
        lines = [head]
        if not rows:
            lines.append("(no rows)")
        else:
            if isinstance(rows[0], dict):
                cols = []
                for r in rows:
                    for k in r:
                        if k not in cols:
                            cols.append(k)
                table = [[_fmt(r.get(c, "")) for c in cols] for r in rows]
                header = [str(c) for c in cols]
            else:
                table = [[_fmt(c) for c in r] for r in rows]
                header = None
            width = [len(h) for h in header] if header else [0] * len(table[0])
            for r in table:
                for i, c in enumerate(r):
                    if i < len(width):
                        width[i] = max(width[i], len(c))
                    else:
                        width.append(len(c))
            if header:
                lines.append("  ".join(h.ljust(width[i]) for i, h in enumerate(header)))
                lines.append("  ".join("-" * width[i] for i in range(len(width))))
            for r in table:
                lines.append("  ".join(c.ljust(width[i]) for i, c in enumerate(r)))
        lines.append("verdict: %s" % self.data.get("verdict", "n/a"))
        lines.append("written: %s" % self.out_path)
        lines.append("=== END ===")
        block = "\n".join(lines)
        print(block)
        return block


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "%.4f" % v if abs(v) < 1e4 else "%.3e" % v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "-"
    return str(v)
