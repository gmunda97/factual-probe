# llama-3.1-8b as the fixed reference map read through every other model
# the reverse map run (reverse_map) builds on its outputs

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import torch

from factgrasp import crossmodel as X, data as D, encoders as E, paths, registry
from experiments.transfer.transfer_all_pairs import ModelSpace, FIELDS

sys.path.insert(0, paths.KG_CODE)

NAME = "reference_map"
RUNS = os.path.join(paths.ROOT, "results", "runs")
PAIRS_PATH = os.path.join(paths.ROOT, "results", "runs", "transfer_all_pairs_all.json")
FLOOR_CSV = os.path.join(paths.ROOT, "final_results", "data", "untrained_floor.csv")
HUB = "meta-llama/Llama-3.1-8B"
HUB2 = "Qwen/Qwen3.5-9B"
THRESHOLDS = {
    "reference_reads_all": "one probe trained once reads the whole atlas: hub-probe MRR at least 0.9 of the native MRR "
           "on 16 of 18 other models (green); 12 to 15 (amber); fewer (red)",
    "reference_beats_weak": "the hub probe reads a weaker model better than that model's own probe: hub-probe MRR above "
           "the native MRR on at least 6 of the models whose native MRR is below the hub's by 0.05 or "
           "more (green); 3 to 5 (amber); fewer (red)",
}


def hub_rows(pairs: list, native: dict, floors: dict, hub: str) -> list:
    out = []
    for r in pairs:
        if r["A"] != hub:
            continue
        m = r["B"]
        out.append({"model": m, "kind": r["kind_B"], "family": r["family_B"], "native_MRR": native[m],
                    "hub_MRR": r["mrr_stitched"], "floor_MRR": floors.get(m), "ratio": r["mrr_stitched"] / native[m],
                    "hub_minus_native": r["mrr_stitched"] - native[m],
                    "hub_minus_floor": (r["mrr_stitched"] - floors[m]) if m in floors else None,
                    "weaker_by_0.05": bool(native[hub] - native[m] >= 0.05), "tau_stored": r["tau"], "cka": r["cka"]})
    return sorted(out, key=lambda x: -x["native_MRR"])


def verdicts(rows: list) -> dict:
    n_ratio = sum(1 for r in rows if r["ratio"] >= 0.9)
    weaker = [r for r in rows if r["weaker_by_0.05"]]
    n_above = sum(1 for r in weaker if r["hub_MRR"] > r["native_MRR"])
    va = "green" if n_ratio >= 16 else "amber" if n_ratio >= 12 else "red"
    vb = "green" if n_above >= 6 else "amber" if n_above >= 3 else "red"
    return {"reference_reads_all": va, "reference_beats_weak": vb, "n_ratio_at_least_0.9": n_ratio, "n_models": len(rows),
            "n_weaker": len(weaker), "n_weaker_read_above_native": n_above,
            "median_ratio": round(statistics.median(r["ratio"] for r in rows), 4),
            "min_ratio": round(min(r["ratio"] for r in rows), 4), "max_ratio": round(max(r["ratio"] for r in rows), 4),
            "ratio_by_model": {r["model"]: round(r["ratio"], 4) for r in rows},
            "hub_above_native": [r["model"] for r in weaker if r["hub_MRR"] > r["native_MRR"]],
            "hub_below_native_among_weaker": [r["model"] for r in weaker if r["hub_MRR"] <= r["native_MRR"]]}


def hub_floors(hubs: list, models: list, floor_rows: dict, run, ridge: float = 1e-2) -> dict:
    from types import SimpleNamespace
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    device = torch.device("cpu")
    spaces = {h: ModelSpace(h, splits, device, SimpleNamespace(temp=0.05, epochs=60)) for h in hubs}
    out = {h: {} for h in hubs}
    for m in models:
        key = registry.config_key({"part": "hub_floor", "model": m})
        if run.done(key):
            for h in hubs:
                out[h][m] = run.get(key)[h]
            continue
        rid = m + "-randinit"; layer, pooling = int(floor_rows[m]["layer"]), floor_rows[m]["pooling"]
        emb = {s: {f: torch.load(E.cache_path(rid, "wikidata", s, f, pooling, layer), map_location="cpu").float()
                   for f in FIELDS} for s in ("train", "test")}
        cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
        proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
        P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
        x_tr = P(emb["train"]["subject"]) + P(emb["train"]["pred_value"])
        x_te = P(emb["test"]["subject"]) + P(emb["test"]["pred_value"])
        twin = SimpleNamespace(x_test=x_te)
        row = {}
        for h in hubs:
            S = X.fit_stitch(x_tr, spaces[h].x_train_cpu, ridge=ridge)
            row[h], _ = spaces[h].stitched(twin, S, splits)
            out[h][m] = row[h]
        run.put(key, row)
        print(f"[{NAME}] hub floor for {m}: " + ", ".join(f"{h.split('/')[-1]} {v:.4f}" for h, v in row.items()), flush=True)
        del emb, cat
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--hub-floors", action="store_true", help="the hub reading each model's untrained twin (control column)")
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    pairs = [v for v in json.load(open(PAIRS_PATH))["rows"].values() if isinstance(v, dict) and "tau" in v]
    native = {r["A"]: r["mrr_native"] for r in pairs}
    floor_rows = {r["model"]: r for r in csv.DictReader(open(FLOOR_CSV))}
    floors = {m: float(r["untrained_MRR"]) for m, r in floor_rows.items()}
    assert max(native, key=native.get) == HUB, f"the hub is not the most readable model: {max(native, key=native.get)}"
    rows = hub_rows(pairs, native, floors, HUB)
    rows2 = hub_rows(pairs, native, floors, HUB2)
    hf = hub_floors([HUB, HUB2], [r["model"] for r in rows if r["model"] in floor_rows], floor_rows, run, args.ridge) if args.hub_floors else {HUB: {}, HUB2: {}}
    for r in rows:
        r["hub_floor_MRR"] = hf[HUB].get(r["model"])
    for r in rows2:
        r["hub_floor_MRR"] = hf[HUB2].get(r["model"])
    for r in rows:
        run.put(registry.config_key({"hub": HUB, "model": r["model"]}), r)
    for r in rows2:
        run.put(registry.config_key({"hub": HUB2, "model": r["model"]}), r)
    v1, v2 = verdicts(rows), verdicts(rows2)
    run.set_summary({"verdicts": {"reference_reads_all": v1["reference_reads_all"], "reference_beats_weak": v1["reference_beats_weak"]},
                     "measured": {"hub": HUB, "hub_native_MRR": round(native[HUB], 4), "hub_floor_MRR": floors.get(HUB),
                                  "reference_reads_all_ratio_at_least_0.9": f"{v1['n_ratio_at_least_0.9']} of {v1['n_models']}",
                                  "reference_beats_weak_weaker_read_above_native": f"{v1['n_weaker_read_above_native']} of {v1['n_weaker']}",
                                  "reference map_llama_hub": v1,
                                  "second_hub": HUB2, "second_hub_native_MRR": round(native[HUB2], 4),
                                  "reference map_qwen_hub": v2,
                                  "hub_floor_MRR_llama_hub": {m: round(v, 4) for m, v in hf[HUB].items()},
                                  "hub_floor_MRR_qwen_hub": {m: round(v, 4) for m, v in hf[HUB2].items()}}})
    run.flush()
    with open(os.path.join(args.csv_dir, "reference_map.csv"), "w") as fh:
        fh.write("model,kind,family,native_MRR,floor_MRR,hub_MRR,ratio,hub_minus_native,hub_minus_floor,weaker_by_0.05,"
                 "hub2_MRR,ratio2,hub2_minus_native,hub_floor_MRR,hub2_floor_MRR,hub,hub2\n")
        by2 = {r["model"]: r for r in rows2}
        for r in rows:
            r2 = by2.get(r["model"])
            fl = "" if r["floor_MRR"] is None else f"{r['floor_MRR']:.6f}"
            hf = "" if r["hub_minus_floor"] is None else f"{r['hub_minus_floor']:+.6f}"
            h2 = "" if r2 is None else f"{r2['hub_MRR']:.6f}"; ra2 = "" if r2 is None else f"{r2['ratio']:.6f}"
            d2 = "" if r2 is None else f"{r2['hub_minus_native']:+.6f}"
            hfl = "" if r.get("hub_floor_MRR") is None else f"{r['hub_floor_MRR']:.6f}"
            hfl2 = "" if r2 is None or r2.get("hub_floor_MRR") is None else f"{r2['hub_floor_MRR']:.6f}"
            fh.write(f"{r['model']},{r['kind']},{r['family']},{r['native_MRR']:.6f},{fl},{r['hub_MRR']:.6f},{r['ratio']:.6f},"
                     f"{r['hub_minus_native']:+.6f},{hf},{int(r['weaker_by_0.05'])},{h2},{ra2},{d2},{hfl},{hfl2},{HUB},{HUB2}\n")
        for h in (HUB, HUB2):
            fl = "" if h not in floors else f"{floors[h]:.6f}"
            fh.write(f"{h},decoder,hub-native,{native[h]:.6f},{fl},,,,,,,,,,,{HUB},{HUB2}\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"hub {HUB}: native {native[HUB]:.4f}, floor {floors.get(HUB)}", flush=True)
    print(f"reference_reads_all  {v1['reference_reads_all']}   hub-probe MRR at least 0.9 of native on {v1['n_ratio_at_least_0.9']} of {v1['n_models']}; "
          f"median ratio {v1['median_ratio']}, min {v1['min_ratio']}, max {v1['max_ratio']}", flush=True)
    print(f"reference_beats_weak  {v1['reference_beats_weak']}   hub above native on {v1['n_weaker_read_above_native']} of {v1['n_weaker']} models weaker by 0.05 or more: "
          + ", ".join(v1["hub_above_native"]), flush=True)
    for r in rows:
        print(f"  {r['model']:45s} native {r['native_MRR']:.4f}  hub {r['hub_MRR']:.4f}  ratio {r['ratio']:.3f}  floor "
              f"{(r['floor_MRR'] if r['floor_MRR'] is not None else float('nan')):.4f}  hub floor "
              f"{(r['hub_floor_MRR'] if r.get('hub_floor_MRR') is not None else float('nan')):.4f}  weaker {int(r['weaker_by_0.05'])}", flush=True)
    print(f"second hub {HUB2}: native {native[HUB2]:.4f}; reference_reads_all-style {v2['n_ratio_at_least_0.9']} of {v2['n_models']} at ratio 0.9 "
          f"(median {v2['median_ratio']}); reference_beats_weak-style {v2['n_weaker_read_above_native']} of {v2['n_weaker']} weaker models read above native", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
