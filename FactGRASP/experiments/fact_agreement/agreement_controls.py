# the 0.977 same model different seed agreement, the untrained twin floor and the object frequency
# terciles for the fact agreement

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from factgrasp import data as D, encoders as E, gpu_guard, layers as L, metrics as M, paths, registry
from experiments.fact_agreement.fact_agreement import EXTRA, within_group_ranks, pair_matrix
from experiments.main_results.all_models import ROSTER

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "agreement_controls"
RUNS = os.path.join(paths.ROOT, "results", "runs")
FIELDS = ("subject", "pred_value", "object")
THRESHOLDS = {
    "seed_ceiling": "same-model two-seed ceiling: median ceiling over models at least 0.7 and every cross-model pair below the "
           "smaller of its two ceilings (green); median 0.55 to 0.7 (amber); below 0.55 (red)",
    "untrained_twin": "the untrained twin does not agree with trained models: median agreement of each twin with the trained "
           "models below 0.15 and every such pair below the cross-model minimum (green); 0.15 to 0.3 (amber); above 0.3 (red)",
    "object_frequency": "the agreement is not the object prior: inside object-frequency terciles, median pairwise agreement at least "
           "0.3 in every tercile (green); in two of three (amber); fewer (red)",
}
FLOOR_TAG = {"tiiuae/falcon-mamba-7b": "falcon_mamba"}


def ladder_rr_seeds(model: str) -> dict:
    slug = E.model_slug(model)
    lad = json.load(open(os.path.join(paths.RUNS, f"ladder_{slug}.json")))["summary"]
    store = torch.load(os.path.join(paths.RUNS, f"ladder_{slug}_rr.pt"), map_location="cpu")
    pre = f"single_infonce|L{lad['best_layer']}|seed"
    return {int(k.split("seed")[-1]): store[k].float().numpy() for k in store if k.startswith(pre)}


def floor_row(model: str) -> dict:
    tag = FLOOR_TAG.get(model, "all")
    for v in json.load(open(os.path.join(RUNS, f"untrained_floor_{tag}.json"))).get("rows", {}).values():
        if isinstance(v, dict) and v.get("model") == model and "untrained_MRR" in v:
            return v
    return {}


def twin_rr(model: str, splits: dict, valid_map, device) -> tuple:
    row = floor_row(model)
    tag = FLOOR_TAG.get(model, "all")
    saved = os.path.join(RUNS, f"untrained_floor_{tag}_rr_{E.model_slug(model)}.pt")
    if int(row.get("untrained_seeds", 1)) == 1 and os.path.exists(saved):
        return torch.load(saved, map_location="cpu").float().numpy(), "saved (untrained floor, one seed)"
    rid = model + "-randinit"; layer, pooling = int(row["layer"]), row["pooling"]
    emb = {s: {f: torch.load(E.cache_path(rid, "wikidata", s, f, pooling, layer), map_location="cpu").float() for f in FIELDS} for s in splits}
    cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
    proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
    P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
    T = {s: {f: P(emb[s][f]) for f in FIELDS} for s in emb}
    x = {s: T[s]["subject"] + T[s]["pred_value"] for s in T}
    names, pool, gold = C.build_unique_object_pool(splits["test"], T["test"]["object"])
    vlists = D.valid_index_lists(splits["test"], names, valid_map)
    m = L.train_single_map(x["train"], T["train"]["pred_value"], T["train"]["object"], x["valid"], T["valid"]["pred_value"],
                           T["valid"]["object"], device, loss="infonce", epochs=60, temp=0.05, seed=0)
    o_hat, _ = L.predict(m, x["test"], T["test"]["pred_value"], device)
    res = M.evaluate(o_hat, pool, gold, valid=vlists, tie="random", tau=0.05, seed=0)
    return res["rr"].float().numpy(), f"retrained seed 0 (untrained floor averaged {row.get('untrained_seeds')} seeds); MRR {float(res['MRR']):.4f}"


def pooled_corr(vectors: dict, groups: np.ndarray) -> tuple:
    names = list(vectors)
    rows, keep_all, gidx = [], None, None
    for n in names:
        r, keep, gidx = within_group_ranks(vectors[n], groups)
        rows.append(r); keep_all = keep if keep_all is None else (keep_all & keep)
    return names, np.stack(rows)[:, keep_all], keep_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    agree_run = json.load(open(os.path.join(RUNS, "fact_agreement_all.json")))
    order = agree_run["summary"]["measured"]["agreement_models"]
    fam = {m: (ROSTER.get(m) or EXTRA.get(m))[0] for m in order}
    agree = {}
    for k, v in agree_run["rows"].items():
        if k.startswith("{") and isinstance(v, dict) and "spearman" in v:
            c = json.loads(k); agree[(c["a"], c["b"])] = v["spearman"]; agree[(c["b"], c["a"])] = v["spearman"]
    cross_min = agree_run["summary"]["measured"]["agreement_min_pairwise"]
    te = D.load_dataset("wikidata", "test")
    rels = te["pred_value"].astype(str).to_numpy()
    ceil, seeds_by = {}, {}
    for m in order:
        rr = ladder_rr_seeds(m)
        if len(rr) < 2:
            continue
        ks = sorted(rr)
        names, R, _ = pooled_corr({f"seed{k}": rr[k] for k in ks}, rels)
        S = pair_matrix(R)
        vals = {f"{ks[i]}-{ks[j]}": float(S[i, j]) for i in range(len(ks)) for j in range(i + 1, len(ks))}
        ceil[m] = float(np.mean(list(vals.values()))); seeds_by[m] = vals
    med_ceil = float(np.median(list(ceil.values())))
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:] if a in ceil and b in ceil]
    below = [(a, b) for a, b in pairs if agree[(a, b)] < min(ceil[a], ceil[b])]
    seed_ceiling_v = "green" if (med_ceil >= 0.7 and len(below) == len(pairs)) else "amber" if med_ceil >= 0.55 else "red"
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    import pandas as pd
    valid_map = D.valid_sets(pd.concat(list(splits.values()), ignore_index=True))
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    trained = {m: ladder_rr_seeds(m)[0] for m in order}
    twins, prov = {}, {}
    for m in order:
        twins[m], prov[m] = twin_rr(m, splits, valid_map, device)
        print(f"[{NAME}] twin of {m}: {prov[m]}", flush=True)
    names, R, _ = pooled_corr({**{f"trained::{m}": trained[m] for m in order}, **{f"twin::{m}": twins[m] for m in order}}, rels)
    S = pair_matrix(R)
    idx = {n: i for i, n in enumerate(names)}
    twin_vs = {t: {m: float(S[idx[f"twin::{t}"], idx[f"trained::{m}"]]) for m in order} for t in order}
    twin_med = {t: float(np.median(list(v.values()))) for t, v in twin_vs.items()}
    all_pairs = [v for d in twin_vs.values() for v in d.values()]
    untrained_v = "green" if (max(twin_med.values()) < 0.15 and max(all_pairs) < cross_min) else "amber" if max(twin_med.values()) <= 0.3 else "red"
    tr = splits["train"]
    cnt = {}
    for r, o in zip(tr["pred_value"].astype(str), tr["object"].astype(str)):
        cnt[(r, o)] = cnt.get((r, o), 0) + 1
    freq = np.array([cnt.get((r, o), 0) for r, o in zip(rels, te["object"].astype(str))])
    terc = np.full(len(te), -1)
    by_rel = {}
    for i, r in enumerate(rels):
        by_rel.setdefault(r, []).append(i)
    for r, ix in by_rel.items():
        if len(ix) < 3:
            continue
        ix = sorted(ix, key=lambda i: (freq[i], i))
        n = len(ix)
        for pos, i in enumerate(ix):
            terc[i] = min(2, pos * 3 // n)
    by_t = {}
    for t in range(3):
        sel = np.where(terc == t)[0]
        names_t, Rt, keep_t = pooled_corr({m: trained[m][sel] for m in order}, rels[sel])
        St = pair_matrix(Rt)
        iu = np.triu_indices(len(order), 1)
        by_t[t + 1] = {"n_rows": int(len(sel)), "n_rows_used": int(Rt.shape[1]), "median": float(np.median(St[iu])), "min": float(St[iu].min()),
                       "max": float(St[iu].max()), "mean_gold_frequency": float(freq[sel].mean()), "share_gold_unseen": float((freq[sel] == 0).mean()), "S": St}
    n_ok = sum(1 for v in by_t.values() if v["median"] >= 0.3)
    frequency_v = "green" if n_ok == 3 else "amber" if n_ok == 2 else "red"
    run.set_summary({"verdicts": {"seed_ceiling": seed_ceiling_v, "untrained_twin": untrained_v, "object_frequency": frequency_v},
                     "measured": {"seed_ceiling_median_ceiling": round(med_ceil, 4), "seed_ceiling_ceiling_min": round(min(ceil.values()), 4), "seed_ceiling_ceiling_max": round(max(ceil.values()), 4),
                                  "seed_ceiling_models_with_seeds": len(ceil), "seed_ceiling_pairs_below_min_ceiling": f"{len(below)} of {len(pairs)}",
                                  "seed_ceiling_ceiling_by_model": {m: round(v, 4) for m, v in ceil.items()},
                                  "seed_ceiling_pairs_at_or_above_ceiling": [f"{a} / {b}" for a, b in pairs if (a, b) not in below],
                                  "untrained_twin_twin_median_by_model": {m: round(v, 4) for m, v in twin_med.items()}, "untrained_twin_twin_median_max": round(max(twin_med.values()), 4),
                                  "untrained_twin_twin_trained_pair_max": round(max(all_pairs), 4), "untrained_twin_cross_model_min": cross_min,
                                  "untrained_twin_pairs_at_or_above_cross_min": [f"{t} twin / {m}" for t, d in twin_vs.items() for m, v in d.items() if v >= cross_min],
                                  "untrained_twin_twin_with_own_trained": {m: round(twin_vs[m][m], 4) for m in order}, "untrained_twin_provenance": prov,
                                  "object_frequency_tercile_medians": {t: round(v["median"], 4) for t, v in by_t.items()}, "object_frequency_tercile_min": {t: round(v["min"], 4) for t, v in by_t.items()},
                                  "object_frequency_tercile_rows": {t: v["n_rows"] for t, v in by_t.items()}, "object_frequency_tercile_mean_gold_frequency": {t: round(v["mean_gold_frequency"], 2) for t, v in by_t.items()},
                                  "object_frequency_tercile_share_gold_unseen_in_training": {t: round(v["share_gold_unseen"], 3) for t, v in by_t.items()},
                                  "object_frequency_relations_with_at_least_3_test_rows": sum(1 for ix in by_rel.values() if len(ix) >= 3)}})
    run.flush()
    with open(os.path.join(args.csv_dir, "agreement_seed_ceiling.csv"), "w") as fh:
        fh.write("row_type,model,family,n_seeds,spearman_0_1,spearman_1_2,spearman_0_2,ceiling,model_b,agreement,min_ceiling,below_ceiling\n")
        for m in order:
            if m in ceil:
                v = seeds_by[m]
                fh.write(f"model,{m},{fam[m]},{len(v) and 3},{v.get('0-1', float('nan')):.6f},{v.get('1-2', float('nan')):.6f},{v.get('0-2', float('nan')):.6f},{ceil[m]:.6f},,,,\n")
        for a, b in pairs:
            fh.write(f"pair,{a},{fam[a]},,,,,,{b},{agree[(a, b)]:.6f},{min(ceil[a], ceil[b]):.6f},{int((a, b) in below)}\n")
    with open(os.path.join(args.csv_dir, "agreement_untrained.csv"), "w") as fh:
        fh.write("row_type,twin_of,trained_model,agreement,twin_median,twin_provenance\n")
        for t in order:
            fh.write(f"summary,{t},,,{twin_med[t]:.6f},{prov[t].replace(',', ';')}\n")
            for m in order:
                fh.write(f"pair,{t},{m},{twin_vs[t][m]:.6f},,\n")
    with open(os.path.join(args.csv_dir, "agreement_object_frequency.csv"), "w") as fh:
        fh.write("row_type,tercile,n_rows,n_rows_used,mean_gold_frequency_in_training,share_gold_unseen,median_spearman,min_spearman,max_spearman,model_a,model_b,spearman\n")
        for t, v in by_t.items():
            fh.write(f"summary,{t},{v['n_rows']},{v['n_rows_used']},{v['mean_gold_frequency']:.3f},{v['share_gold_unseen']:.4f},{v['median']:.6f},{v['min']:.6f},{v['max']:.6f},,,\n")
            for i, a in enumerate(order):
                for j, b in enumerate(order):
                    if j > i:
                        fh.write(f"pair,{t},,,,,,,,{a},{b},{v['S'][i, j]:.6f}\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"seed_ceiling  {seed_ceiling_v}   median same-model two-seed ceiling {med_ceil:.4f} (min {min(ceil.values()):.4f}, max {max(ceil.values()):.4f}) over {len(ceil)} models; "
          f"cross-model pairs below the smaller ceiling {len(below)} of {len(pairs)}", flush=True)
    print(f"untrained_twin  {untrained_v}   twin medians {min(twin_med.values()):.4f} to {max(twin_med.values()):.4f}; largest twin-trained pair {max(all_pairs):.4f} against the cross-model minimum {cross_min:.4f}", flush=True)
    print(f"object_frequency  {frequency_v}   tercile medians " + ", ".join(f"T{t} {v['median']:.4f} (min {v['min']:.4f}, {v['n_rows']} rows, gold seen in training on {100 * (1 - v['share_gold_unseen']):.0f} percent)" for t, v in by_t.items()), flush=True)
    for m in order:
        print(f"  {m:45s} ceiling {ceil.get(m, float('nan')):.3f}  twin median {twin_med[m]:.3f}  twin with own trained {twin_vs[m][m]:.3f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
