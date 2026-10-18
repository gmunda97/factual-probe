# small io helpers shared by transfer_summary and selected_vs_last_layer (json and csv writing, hashes)

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

from factgrasp import paths

RAW = os.path.join(paths.ROOT, "results", "summaries")
FINAL = os.path.join(paths.ROOT, "final_results", "summaries")
RESULT_COLS = ["queue_id", "condition_id", "model", "checkpoint", "dataset", "split", "seed", "layer", "selection_role",
               "metric", "value", "n", "ci_low", "ci_high", "status", "source_path", "notes"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed_inputs(paths_list) -> list:
    out = []
    for p in paths_list:
        p = os.path.abspath(p)
        out.append({"path": p, "sha256": sha256(p) if os.path.isfile(p) else None, "bytes": os.path.getsize(p) if os.path.isfile(p) else None})
    return out


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def write_results_csv(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in RESULT_COLS})


def append_job(record: dict) -> None:
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "jobs.jsonl"), "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def update_status(**fields) -> dict:
    p = os.path.join(RAW, "status.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(fields); cur["updated_at_utc"] = utc_now()
    write_json(p, cur)
    write_json(os.path.join(FINAL, "status.json"), cur)
    return cur


def manifest(root: str, out_path: str) -> None:
    lines = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if os.path.abspath(p) == os.path.abspath(out_path):
                continue
            lines.append(f"{sha256(p)}  {os.path.relpath(p, root)}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(sorted(lines)) + "\n")
