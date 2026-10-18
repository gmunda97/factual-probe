# reverse map of appendix d: send the transferred prediction back into model B's own candidate space
# with a second alignment fitted on composed vectors (median 64 percent) or on object vectors (80 percent)

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

from factgrasp import crossmodel as X, data as D, encoders as E, metrics as M, paths, registry
from experiments.transfer.transfer_all_pairs import ModelSpace, FIELDS

NAME = "reverse_map"
RUNS = os.path.join(paths.ROOT, "results", "runs")
DATA = os.path.join(paths.ROOT, "final_results", "data")
HUB, HUB2 = "meta-llama/Llama-3.1-8B", "Qwen/Qwen3.5-9B"
ROSTER = ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base",
          "Qwen/Qwen3.5-9B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct", "answerdotai/ModernBERT-base",
          "google/gemma-4-12B", "google/gemma-4-E2B", "google/gemma-4-E4B", "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-8b",
          "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
          "mistralai/Ministral-3-8B-Instruct-2512-BF16"]
MEAN_POOLED = {"answerdotai/ModernBERT-base": "answerdotai/ModernBERT-base_mean"}
THRESHOLDS = {
    "pool_matched": "the hub's advantage survives pool matching: under (a) hub-pool-matched MRR at least 0.9 of native on 16 of "
           "18 and above native on at least 6 of the weaker models, and under (b) the hub's gain above floor at least "
           "the native gain above floor on at least 6 of the weaker models (green); one of the two readings (amber); "
           "neither (red; the 'better than its own probe' clause is withdrawn)",
}


def five_seed_native() -> dict:
    return {r["model"]: float(r["MRR_single_infonce_5seed"]) for r in csv.DictReader(open(os.path.join(DATA, "seed_counts.csv")))}


def floor_rows_csv() -> dict:
    out = {}
    for r in csv.DictReader(open(os.path.join(DATA, "untrained_floor.csv"))):
        if "superseded" in r["status"]:
            continue
        out[r["model"]] = r
    return out


def twin_space(model_id: str, floor_rows: dict, splits: dict) -> SimpleNamespace:
    rid = model_id + "-randinit"; layer, pooling = int(floor_rows["layer"]), floor_rows["pooling"]
    emb = {s: {f: torch.load(E.cache_path(rid, "wikidata", s, f, pooling, layer), map_location="cpu").float()
               for f in FIELDS} for s in ("train", "test")}
    cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
    proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
    P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
    return SimpleNamespace(x_train_cpu=P(emb["train"]["subject"]) + P(emb["train"]["pred_value"]),
                           x_test=P(emb["test"]["subject"]) + P(emb["test"]["pred_value"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--csv-dir", default=DATA)
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    device = torch.device("cpu")
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    native5, floor_rows = five_seed_native(), floor_rows_csv()
    ref_summary = json.load(open(os.path.join(RUNS, "reference_map_all.json")))["summary"]["measured"]
    hub_floor_prev = {HUB: ref_summary.get("hub_floor_MRR_llama_hub", {}), HUB2: ref_summary.get("hub_floor_MRR_qwen_hub", {})}
    hubs = {h: ModelSpace(h, splits, device, SimpleNamespace(temp=0.05, epochs=60)) for h in (HUB, HUB2)}
    rows = []
    for m in ROSTER:
        key = registry.config_key({"model": m, "ridge": args.ridge})
        if run.done(key):
            rows.append(run.get(key)); continue
        t0 = time.time()
        mid = MEAN_POOLED.get(m, m)
        sp = ModelSpace(mid, splits, device, SimpleNamespace(temp=0.05, epochs=60))
        row = {"model": m, "model_id_used": mid, "layer": sp.layer, "pooling": sp.pooling, "native_MRR": native5.get(m),
               "native_MRR_transfer_seed0": sp.mrr_native, "native_floor": float(floor_rows[m]["untrained_MRR"]) if m in floor_rows else None,
               "native_source": "five-seed atlas row" if m in native5 else "none",
               "hubs": {}}
        if m in MEAN_POOLED:
            row["native_MRR"] = float(floor_rows[m]["trained_MRR"]); row["native_source"] = "mean-pooled ladder three seeds (the atlas row)"
        twin = twin_space(m, floor_rows[m], splits) if (m in floor_rows and (m in MEAN_POOLED or m not in hub_floor_prev[HUB])) else None
        for h, hs in hubs.items():
            S_fwd = X.fit_stitch(sp.x_train_cpu, hs.x_train_cpu, ridge=args.ridge)
            S_rev = X.fit_stitch(hs.x_train_cpu, sp.x_train_cpu, ridge=args.ridge)
            S_rev_obj = X.fit_stitch(hs.o_train.cpu(), sp.o_train.cpu(), ridge=args.ridge)
            mrr_hub, v = hs.stitched(sp, S_fwd, splits)
            v = v.cpu()
            matched = M.evaluate(v @ S_rev, sp.pool_cpu, sp.gold, valid=sp.vlists(splits), tie="random", seed=0)
            matched_obj = M.evaluate(v @ S_rev_obj, sp.pool_cpu, sp.gold, valid=sp.vlists(splits), tie="random", seed=0)
            if twin is not None:
                S_tw = X.fit_stitch(twin.x_train_cpu, hs.x_train_cpu, ridge=args.ridge)
                hub_floor, _ = hs.stitched(twin, S_tw, splits)
            else:
                hub_floor = hub_floor_prev[h].get(m)
            row["hubs"][h] = {"hub_native_MRR": hs.mrr_native, "hub_native_MRR_5seed": native5.get(h), "hub_MRR": mrr_hub, "hub_floor": hub_floor,
                              "hub_matched_own_pool": float(matched["MRR"]), "hub_matched_own_pool_Hits1": float(matched["Hits@1"]),
                              "hub_matched_own_pool_objfit": float(matched_obj["MRR"])}
            print(f"[{NAME}] {m} via {h.split('/')[-1]}: native5 {row['native_MRR']:.4f} (s7 {sp.mrr_native:.4f}) floor {row['native_floor']:.4f} | "
                  f"hub {mrr_hub:.4f} hub floor {hub_floor:.4f} matched {matched['MRR']:.4f} (objfit {matched_obj['MRR']:.4f})", flush=True)
        row["seconds"] = round(time.time() - t0)
        run.put(key, row); rows.append(row)
        del sp
    def readings(h):
        hub_nat = native5[h]
        others = [r for r in rows if r["model"] != h]
        weaker = [r for r in others if hub_nat - r["native_MRR"] >= 0.05]
        n_ratio = sum(1 for r in others if r["hubs"][h]["hub_matched_own_pool"] / r["native_MRR"] >= 0.9)
        n_above = sum(1 for r in weaker if r["hubs"][h]["hub_matched_own_pool"] > r["native_MRR"])
        n_gain = sum(1 for r in weaker if (r["hubs"][h]["hub_MRR"] - r["hubs"][h]["hub_floor"]) >= (r["native_MRR"] - r["native_floor"]))
        a_ok = n_ratio >= 16 and n_above >= 6
        b_ok = n_gain >= 6
        n_w1a = sum(1 for r in others if r["hubs"][h]["hub_MRR"] / r["native_MRR"] >= 0.9)
        n_w1b = sum(1 for r in weaker if r["hubs"][h]["hub_MRR"] > r["native_MRR"])
        return {"n_others": len(others), "n_weaker": len(weaker), "a_ratio_at_least_0.9": n_ratio, "a_above_native_among_weaker": n_above,
                "a_holds": a_ok, "b_gain_at_least_native_gain_among_weaker": n_gain, "b_holds": b_ok,
                "median_matched_ratio": round(statistics.median(r["hubs"][h]["hub_matched_own_pool"] / r["native_MRR"] for r in others), 4),
                "median_hub_gain": round(statistics.median(r["hubs"][h]["hub_MRR"] - r["hubs"][h]["hub_floor"] for r in others), 4),
                "median_native_gain": round(statistics.median(r["native_MRR"] - r["native_floor"] for r in others), 4),
                "reference_reads_all_regenerated_ratio_at_least_0.9": n_w1a, "reference_beats_weak_regenerated_above_native": n_w1b,
                "weaker_models": [r["model"] for r in weaker]}
    r1, r2 = readings(HUB), readings(HUB2)
    verdict = "green" if (r1["a_holds"] and r1["b_holds"]) else "amber" if (r1["a_holds"] or r1["b_holds"]) else "red"
    run.set_summary({"verdicts": {"pool_matched": verdict},
                     "measured": {"pool_matched_reading_a": f"matched at least 0.9 of native on {r1['a_ratio_at_least_0.9']} of {r1['n_others']}; above native on "
                                                   f"{r1['a_above_native_among_weaker']} of {r1['n_weaker']} weaker",
                                  "pool_matched_reading_b": f"hub gain above floor at least native gain on {r1['b_gain_at_least_native_gain_among_weaker']} of {r1['n_weaker']} weaker",
                                  "pool_matched_llama_hub": r1, "pool_matched_qwen_hub": r2, "hub": HUB, "second_hub": HUB2,
                                  "reference map_regenerated_llama": {"reference_reads_all": f"{r1['reference_reads_all_regenerated_ratio_at_least_0.9']} of {r1['n_others']}", "reference_beats_weak": f"{r1['reference_beats_weak_regenerated_above_native']} of {r1['n_weaker']}"},
                                  "reference map_regenerated_qwen": {"reference_reads_all": f"{r2['reference_reads_all_regenerated_ratio_at_least_0.9']} of {r2['n_others']}", "reference_beats_weak": f"{r2['reference_beats_weak_regenerated_above_native']} of {r2['n_weaker']}"},
                                  "native_source": "five-seed atlas row (seed_counts.csv); ModernBERT-base the mean-pooled ladder"}})
    run.flush()
    f6 = lambda v: "" if v is None else f"{v:.6f}"
    with open(os.path.join(args.csv_dir, "reverse_map.csv"), "w") as fh:
        fh.write("model,pooling,native,native_floor,hub,hub_floor,hub_matched_own_pool,ratio_matched,hub_gain_above_floor,native_gain_above_floor,"
                 "weaker_by_0.05,hub2,hub2_floor,hub2_matched_own_pool,ratio2_matched,hub2_gain_above_floor,hub_matched_own_pool_objfit,"
                 "hub2_matched_own_pool_objfit,native_transfer_seed0,native_source,hub_id,hub2_id\n")
        hub_nat = native5[HUB]
        for r in sorted(rows, key=lambda r: -r["native_MRR"]):
            a, b = r["hubs"][HUB], r["hubs"][HUB2]
            fh.write(f"{r['model']},{r['pooling']},{f6(r['native_MRR'])},{f6(r['native_floor'])},{f6(a['hub_MRR'])},{f6(a['hub_floor'])},"
                     f"{f6(a['hub_matched_own_pool'])},{f6(a['hub_matched_own_pool'] / r['native_MRR'])},{f6(a['hub_MRR'] - a['hub_floor'])},"
                     f"{f6(r['native_MRR'] - r['native_floor'])},{int(hub_nat - r['native_MRR'] >= 0.05)},{f6(b['hub_MRR'])},{f6(b['hub_floor'])},"
                     f"{f6(b['hub_matched_own_pool'])},{f6(b['hub_matched_own_pool'] / r['native_MRR'])},{f6(b['hub_MRR'] - b['hub_floor'])},"
                     f"{f6(a['hub_matched_own_pool_objfit'])},{f6(b['hub_matched_own_pool_objfit'])},{f6(r['native_MRR_transfer_seed0'])},"
                     f"{r['native_source']},{HUB},{HUB2}\n")
    with open(os.path.join(args.csv_dir, "reference_map.csv"), "w") as fh:
        fh.write("model,kind,family,native_MRR,floor_MRR,hub_MRR,ratio,hub_minus_native,hub_minus_floor,weaker_by_0.05,"
                 "hub2_MRR,ratio2,hub2_minus_native,hub_floor_MRR,hub2_floor_MRR,hub,hub2,native_MRR_transfer_seed0,native_source,pooling\n")
        for r in sorted(rows, key=lambda r: -r["native_MRR"]):
            if r["model"] == HUB:
                continue
            a, b = r["hubs"][HUB], r["hubs"][HUB2]
            kind = "encoder" if "ModernBERT" in r["model"] else "decoder"
            fam = r["model"].split("/")[-1].rsplit("-", 1)[0]
            fh.write(f"{r['model']},{kind},{fam},{f6(r['native_MRR'])},{f6(r['native_floor'])},{f6(a['hub_MRR'])},{f6(a['hub_MRR'] / r['native_MRR'])},"
                     f"{a['hub_MRR'] - r['native_MRR']:+.6f},{a['hub_MRR'] - r['native_floor']:+.6f},{int(hub_nat - r['native_MRR'] >= 0.05)},"
                     f"{f6(b['hub_MRR'])},{f6(b['hub_MRR'] / r['native_MRR'])},{b['hub_MRR'] - r['native_MRR']:+.6f},{f6(a['hub_floor'])},{f6(b['hub_floor'])},"
                     f"{HUB},{HUB2},{f6(r['native_MRR_transfer_seed0'])},{r['native_source']},{r['pooling']}\n")
        for h in (HUB, HUB2):
            rr = next(r for r in rows if r["model"] == h)
            fh.write(f"{h},decoder,hub-native,{f6(rr['native_MRR'])},{f6(rr['native_floor'])},,,,,,,,,,,{HUB},{HUB2},{f6(rr['native_MRR_transfer_seed0'])},{rr['native_source']},{rr['pooling']}\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"pool_matched  {verdict}   (a) matched at least 0.9 of native on {r1['a_ratio_at_least_0.9']} of {r1['n_others']}, above native on "
          f"{r1['a_above_native_among_weaker']} of {r1['n_weaker']} weaker (median matched ratio {r1['median_matched_ratio']}); "
          f"(b) hub gain above floor at least native gain on {r1['b_gain_at_least_native_gain_among_weaker']} of {r1['n_weaker']} "
          f"(median hub gain {r1['median_hub_gain']}, native gain {r1['median_native_gain']})", flush=True)
    print(f"Qwen3.5-9B hub: (a) {r2['a_ratio_at_least_0.9']} of {r2['n_others']} and {r2['a_above_native_among_weaker']} of {r2['n_weaker']}; "
          f"(b) {r2['b_gain_at_least_native_gain_among_weaker']} of {r2['n_weaker']}", flush=True)
    print(f"reference map regenerated from the five-seed source: Llama hub reference_reads_all {r1['reference_reads_all_regenerated_ratio_at_least_0.9']} of {r1['n_others']}, "
          f"reference_beats_weak {r1['reference_beats_weak_regenerated_above_native']} of {r1['n_weaker']}; Qwen hub {r2['reference_reads_all_regenerated_ratio_at_least_0.9']} of {r2['n_others']}, "
          f"{r2['reference_beats_weak_regenerated_above_native']} of {r2['n_weaker']}", flush=True)
    for r in sorted(rows, key=lambda r: -r["native_MRR"]):
        a = r["hubs"][HUB]
        print(f"  {r['model']:45s} native {r['native_MRR']:.4f} floor {r['native_floor']:.4f} | hub {a['hub_MRR']:.4f} hub floor {a['hub_floor']:.4f} "
              f"matched {a['hub_matched_own_pool']:.4f} ({a['hub_matched_own_pool'] / r['native_MRR']:.3f}) objfit {a['hub_matched_own_pool_objfit']:.4f} | "
              f"gain hub {a['hub_MRR'] - a['hub_floor']:+.3f} native {r['native_MRR'] - r['native_floor']:+.3f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
