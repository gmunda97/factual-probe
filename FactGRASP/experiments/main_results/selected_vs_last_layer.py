# table 5, the valid selected layer vs the last layer for every model from the saved ladder json
# run: python -m experiments.main_results.selected_vs_last_layer --table

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

from factgrasp import data as D, encoders as E, gpu_guard, layers as L, metrics as M, paths, registry
from factgrasp.io_utils import RAW, FINAL, write_results_csv

sys.path.insert(0, paths.KG_CODE)
import common as C

FIELDS = ("subject", "pred_value", "object")
ORACLE_DIR = os.path.join(RAW, "layer_oracle")
PROTO = json.load(open(os.path.join(FINAL, "layer_protocol.json")))
GRID = PROTO["layer_grid"]


def ladder_summary(model: str) -> dict:
    return json.load(open(os.path.join(paths.RUNS, f"ladder_{E.model_slug(model)}.json")))["summary"]


def ladder_rows(model: str, row: str, layer: int) -> list:
    blob = json.load(open(os.path.join(paths.RUNS, f"ladder_{E.model_slug(model)}.json")))
    out = []
    for k, v in blob.get("rows", {}).items():
        if not k.startswith("{") or not isinstance(v, dict) or "MRR" not in v:
            continue
        c = json.loads(k)
        if c.get("stage") == 2 and c.get("layer") == layer and (v.get("row") or c.get("row")) == row and "seed" in c:
            out.append((c["seed"], float(v["MRR"])))
    return sorted(out)


def table() -> int:
    rows, results = [], []
    for m, grid in GRID.items():
        s = ladder_summary(m)
        best, last = int(s["best_layer"]), int(s["n_layers"])
        sel = ladder_rows(m, "single_infonce", best); fin = ladder_rows(m, "single_infonce_last_layer", last)
        if not fin:
            fin = ladder_rows(m, "single_infonce", last)
        v_sel = statistics.mean(x for _, x in sel) if sel else s.get("MRR_single_infonce")
        v_fin = statistics.mean(x for _, x in fin) if fin else s.get("MRR_single_infonce_last_layer")
        rows.append({"model": m, "checkpoint": m, "pooling": s.get("pooling"), "n_layers": last, "validation_selected_layer": best, "final_layer": last,
                     "selected_equals_final": int(best == last), "validation_selected_test_mrr": v_sel, "final_layer_test_mrr": v_fin,
                     "selected_minus_final": (v_sel - v_fin) if (v_sel is not None and v_fin is not None) else None,
                     "seeds_selected": ";".join(str(k) for k, _ in sel), "seeds_final": ";".join(str(k) for k, _ in fin),
                     "grid_cached": ";".join(str(x) for x in grid), "source_path": os.path.join(paths.RUNS, f"ladder_{E.model_slug(m)}.json")})
        for k, v in sel:
            results.append({"queue_id": "layer summary", "condition_id": f"table::{m}::selected", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "test", "seed": k, "layer": best,
                            "selection_role": "validation_selected", "metric": "MRR", "value": v, "n": 4243, "status": "completed", "source_path": rows[-1]["source_path"]})
        for k, v in fin:
            results.append({"queue_id": "layer summary", "condition_id": f"table::{m}::final", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "test", "seed": k, "layer": last,
                            "selection_role": "final_layer", "metric": "MRR", "value": v, "n": 4243, "status": "completed", "source_path": rows[-1]["source_path"]})
    with open(os.path.join(FINAL, "selected_vs_last_layer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    write_results_csv(os.path.join(FINAL, "layer_results_table.csv"), results)
    d = [r["selected_minus_final"] for r in rows if r["selected_minus_final"] is not None and not r["selected_equals_final"]]
    print(f"[q3 table] {len(rows)} models; selected equals final on {sum(r['selected_equals_final'] for r in rows)}; among the rest selected minus final mean "
          f"{statistics.mean(d):+.4f}, median {statistics.median(d):+.4f}, min {min(d):+.4f}, max {max(d):+.4f}, positive on {sum(x > 0 for x in d)} of {len(d)}", flush=True)
    return 0


def oracle(models: list, seeds: list, batch_note: str) -> int:
    device = gpu_guard.acquire(min_free_gb=10)
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    import pandas as pd
    valid_map = D.valid_sets(pd.concat(list(splits.values()), ignore_index=True))
    names0, _, _ = C.build_unique_object_pool(splits["test"], torch.zeros(len(splits["test"]), 1))
    vlists = D.valid_index_lists(splits["test"], names0, valid_map)
    for m in models:
        slug = E.model_slug(m); s = ladder_summary(m); pooling = s.get("pooling") or "last"
        run = registry.Run("layer_oracle", slug, config={"model": m, "grid": GRID[m], "seeds": seeds, "recipe": "ladder stage 2"}, thresholds={}, out_dir=ORACLE_DIR)
        for layer in GRID[m]:
            for seed in seeds:
                key = registry.config_key({"model": m, "layer": layer, "seed": seed})
                if run.done(key):
                    continue
                t0 = time.time()
                try:
                    emb = {sp: {f: torch.load(E.cache_path(m, "wikidata", sp, f, pooling, layer), map_location="cpu").float() for f in FIELDS} for sp in splits}
                except Exception as exc:
                    run.put(key, {"model": m, "layer": layer, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}); run.flush()
                    print(f"[q3 oracle] {m} L{layer} seed {seed}: cache load failed ({exc})", flush=True); continue
                cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
                proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
                P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
                T = {sp: {f: P(emb[sp][f]) for f in FIELDS} for sp in emb}
                x = {sp: T[sp]["subject"] + T[sp]["pred_value"] for sp in T}
                names, pool, gold = C.build_unique_object_pool(splits["test"], T["test"]["object"])
                vnames, vpool, vgold = C.build_unique_object_pool(splits["valid"], T["valid"]["object"])
                vvl = D.valid_index_lists(splits["valid"], vnames, valid_map)
                probe = L.train_single_map(x["train"], T["train"]["pred_value"], T["train"]["object"], x["valid"], T["valid"]["pred_value"], T["valid"]["object"],
                                           device, loss="infonce", epochs=60, temp=0.05, seed=seed)
                o_te, _ = L.predict(probe, x["test"], T["test"]["pred_value"], device)
                o_va, _ = L.predict(probe, x["valid"], T["valid"]["pred_value"], device)
                r_te = M.evaluate(o_te, pool, gold, valid=vlists, tie="random", seed=seed)
                r_va = M.evaluate(o_va, vpool, vgold, valid=vvl, tie="random", seed=seed)
                row = {"model": m, "layer": layer, "seed": seed, "status": "completed", "test_MRR": float(r_te["MRR"]), "test_Hits1": float(r_te["Hits@1"]),
                       "valid_MRR": float(r_va["MRR"]), "valid_pool": int(vpool.shape[0]), "test_pool": int(pool.shape[0]), "seconds": round(time.time() - t0, 1)}
                run.put(key, row); run.flush()
                print(f"[q3 oracle] {m} L{layer} seed {seed}: valid {row['valid_MRR']:.4f} test {row['test_MRR']:.4f} ({row['seconds']}s)", flush=True)
                del probe, emb, T, x
                if gpu_guard.check_yield():
                    gpu_guard.yield_and_exit(f"q3 oracle after {m} L{layer} seed {seed}")
        print(f"[q3 oracle] {m}: grid complete", flush=True)
    return 0


def collect(seeds: list) -> int:
    out, results, coverage = [], [], []
    for m, grid in GRID.items():
        s = ladder_summary(m); best, last = int(s["best_layer"]), int(s["n_layers"])
        p = os.path.join(ORACLE_DIR, f"layer_oracle_{E.model_slug(m)}.json")
        fits = {}
        if os.path.exists(p):
            for v in json.load(open(p)).get("rows", {}).values():
                if isinstance(v, dict) and v.get("status") == "completed":
                    fits.setdefault(int(v["layer"]), {})[int(v["seed"])] = v
        done_layers = sorted(l for l, d in fits.items() if all(k in d for k in seeds))
        coverage.append({"model": m, "grid": ";".join(map(str, grid)), "n_grid": len(grid), "n_layers_completed": len(done_layers), "layers_completed": ";".join(map(str, done_layers)),
                         "complete": int(len(done_layers) == len(grid))})
        if not done_layers:
            out.append({"model": m, "checkpoint": m, "grid": ";".join(map(str, grid)), "validation_selected_layer": best, "final_layer": last, "test_oracle_layer": "", "full_fit_validation_layer": "",
                        "validation_selected_test_mrr": "", "final_layer_test_mrr": "", "test_oracle_mrr": "", "full_fit_validation_test_mrr": "", "selected_regret": "", "final_regret": "",
                        "seeds": ";".join(map(str, seeds)), "status": "not_run" if not os.path.exists(p) else "no_complete_layer"})
            continue
        mean_te = {l: statistics.mean(fits[l][k]["test_MRR"] for k in seeds) for l in done_layers}
        mean_va = {l: statistics.mean(fits[l][k]["valid_MRR"] for k in seeds) for l in done_layers}
        orc = max(done_layers, key=lambda l: mean_te[l]); fvl = max(done_layers, key=lambda l: mean_va[l])
        status = "completed" if len(done_layers) == len(grid) else "partial_grid"
        row = {"model": m, "checkpoint": m, "grid": ";".join(map(str, grid)), "validation_selected_layer": best, "final_layer": last, "test_oracle_layer": orc, "full_fit_validation_layer": fvl,
               "validation_selected_test_mrr": mean_te.get(best), "final_layer_test_mrr": mean_te.get(last), "test_oracle_mrr": mean_te[orc], "full_fit_validation_test_mrr": mean_te[fvl],
               "selected_regret": (mean_te[orc] - mean_te[best]) if best in mean_te else "", "final_regret": (mean_te[orc] - mean_te[last]) if last in mean_te else "",
               "seeds": ";".join(map(str, seeds)), "status": status, "layers_completed": len(done_layers), "n_grid": len(grid),
               "ladder_selected_test_mrr": s.get("MRR_single_infonce"), "ladder_final_test_mrr": s.get("MRR_single_infonce_last_layer")}
        out.append(row)
        for l in done_layers:
            for k in seeds:
                v = fits[l][k]
                role = "test_oracle" if l == orc else ("validation_selected" if l == best else ("final_layer" if l == last else "grid"))
                results.append({"queue_id": "layer summary", "condition_id": f"oracle::{m}::L{l}", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "test", "seed": k, "layer": l,
                                "selection_role": role, "metric": "MRR", "value": v["test_MRR"], "n": 4243, "status": "completed", "source_path": p})
                results.append({"queue_id": "layer summary", "condition_id": f"oracle::{m}::L{l}", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "valid", "seed": k, "layer": l,
                                "selection_role": role, "metric": "MRR", "value": v["valid_MRR"], "n": 4242, "status": "completed", "source_path": p})
    cols = ["model", "checkpoint", "grid", "validation_selected_layer", "final_layer", "test_oracle_layer", "validation_selected_test_mrr", "final_layer_test_mrr", "test_oracle_mrr",
            "selected_regret", "final_regret", "seeds", "status", "full_fit_validation_layer", "full_fit_validation_test_mrr", "layers_completed", "n_grid", "ladder_selected_test_mrr", "ladder_final_test_mrr"]
    with open(os.path.join(FINAL, "layer_oracle.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in out]
    with open(os.path.join(FINAL, "layer_cache_inventory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(coverage[0].keys())); w.writeheader(); [w.writerow(c) for c in coverage]
    write_results_csv(os.path.join(FINAL, "layer_oracle_results.csv"), results)
    done = [r for r in out if r["status"] == "completed"]
    if done:
        sr = [r["selected_regret"] for r in done if r["selected_regret"] != ""]; fr = [r["final_regret"] for r in done if r["final_regret"] != ""]
        print(f"[q3 collect] {len(done)} of {len(out)} models complete; selected regret median {statistics.median(sr):.4f} max {max(sr):.4f}, zero on {sum(x == 0 for x in sr)}; "
              f"final regret median {statistics.median(fr):.4f} max {max(fr):.4f}; oracle layer equals selected on {sum(r['test_oracle_layer'] == r['validation_selected_layer'] for r in done)}, "
              f"full-fit validation layer equals oracle on {sum(r['full_fit_validation_layer'] == r['test_oracle_layer'] for r in done)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true"); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--collect", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(FINAL, exist_ok=True); os.makedirs(ORACLE_DIR, exist_ok=True)
    if args.table:
        table()
    if args.oracle:
        oracle([m.strip() for m in args.models.split(",") if m.strip()], seeds, "")
    if args.collect:
        collect(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
