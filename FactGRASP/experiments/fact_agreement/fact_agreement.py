# figure 4b and the 0.549 median
# for every pair of the 20 models (19 + falcon-mamba) the spearman between per fact reciprocal ranks,
# ranks taken inside each relation then pooled, with a permutation null

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch
from scipy.stats import rankdata

from factgrasp import data as D, encoders as E, paths, registry
from experiments.main_results.all_models import ROSTER

NAME = "fact_agreement"
RUNS = os.path.join(paths.ROOT, "results", "runs")
EXTRA = {"tiiuae/falcon-mamba-7b": ("FalconMamba", 7.27, "base")}
ATLAS = [m for m in ROSTER if m != "Qwen/Qwen3.5-27B-GPTQ-Int4"]
THRESHOLDS = {
    "agreement": "the facts that are hard are the same facts in every model: median pairwise within-relation Spearman at "
           "least 0.3 and every pair above the null's 99th percentile (green); median 0.15 to 0.3 with every pair above "
           "the null (amber); lower, or any pair inside the null (red)",
}


def wikidata_rr(model: str) -> torch.Tensor:
    slug = E.model_slug(model)
    lad = json.load(open(os.path.join(paths.RUNS, f"ladder_{slug}.json")))["summary"]
    store = torch.load(os.path.join(paths.RUNS, f"ladder_{slug}_rr.pt"), map_location="cpu")
    return store[f"single_infonce|L{lad['best_layer']}|seed0"].float()


def within_group_ranks(values: np.ndarray, groups: np.ndarray) -> tuple:
    out = np.full(values.shape, np.nan)
    idx_by = {}
    for i, g in enumerate(groups):
        idx_by.setdefault(g, []).append(i)
    keep = np.zeros(len(values), dtype=bool)
    for g, idx in idx_by.items():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        out[idx] = rankdata(values[idx]) / (len(idx) + 1.0)
        keep[idx] = True
    return out, keep, idx_by


def pair_matrix(R: np.ndarray) -> np.ndarray:
    return np.corrcoef(R)


def null_matrix(R: np.ndarray, groups_idx: dict, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M, N = R.shape
    draws = np.empty((n_draws, M, M))
    for d in range(n_draws):
        perm = np.arange(N)
        for idx in groups_idx.values():
            if len(idx) >= 2:
                idx = np.array(idx); perm[idx] = idx[rng.permutation(len(idx))]
        Rp = R[:, perm]
        A = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True); B = (Rp - Rp.mean(1, keepdims=True)) / Rp.std(1, keepdims=True)
        draws[d] = A @ B.T / N
    return np.percentile(draws, 99, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    models = ATLAS + list(EXTRA)
    fam = {m: (ROSTER.get(m) or EXTRA.get(m))[0] for m in models}
    order = sorted(models, key=lambda m: (fam[m], (ROSTER.get(m) or EXTRA.get(m))[1]))
    te = D.load_dataset("wikidata", "test")
    rels = te["pred_value"].astype(str).to_numpy()
    rows, keep_all = [], None
    for m in order:
        rr = wikidata_rr(m).numpy()
        r, keep, groups_idx = within_group_ranks(rr, rels)
        rows.append(r); keep_all = keep if keep_all is None else (keep_all & keep)
    R = np.stack(rows)[:, keep_all]
    groups_kept = {g: [i for i in idx] for g, idx in groups_idx.items() if len(idx) >= 2}
    kept_pos = {old: new for new, old in enumerate(np.where(keep_all)[0])}
    groups_kept = {g: [kept_pos[i] for i in idx if i in kept_pos] for g, idx in groups_kept.items()}
    S = pair_matrix(R)
    N99 = null_matrix(R, groups_kept, args.draws, args.seed)
    iu = np.triu_indices(len(order), 1)
    vals, nulls = S[iu], N99[iu]
    med, lo = float(np.median(vals)), float(vals.min())
    above = int((vals > nulls).sum()); n_pairs = len(vals)
    agreement_v = "green" if (med >= 0.3 and above == n_pairs) else "amber" if (med >= 0.15 and above == n_pairs) else "red"
    same = [S[i, j] for i, j in zip(*iu) if fam[order[i]] == fam[order[j]]]
    cross = [S[i, j] for i, j in zip(*iu) if fam[order[i]] != fam[order[j]]]
    run.set_summary({"verdicts": {"agreement": agreement_v},
                     "measured": {"agreement_median_pairwise_spearman": round(med, 4), "agreement_min_pairwise": round(lo, 4), "agreement_max_pairwise": round(float(vals.max()), 4),
                                  "agreement_pairs_above_null99": f"{above} of {n_pairs}", "agreement_null99_median": round(float(np.median(nulls)), 4),
                                  "agreement_null99_max": round(float(nulls.max()), 4), "agreement_n_rows_used": int(R.shape[1]), "agreement_n_relations_used": len(groups_kept),
                                  "agreement_same_family_median": round(float(np.median(same)), 4) if same else None,
                                  "agreement_cross_family_median": round(float(np.median(cross)), 4),
                                  "agreement_models": order,
                                  "draws": args.draws}})
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j <= i:
                continue
            run.put(registry.config_key({"graph": "wikidata", "a": a, "b": b}), {"spearman": float(S[i, j]), "null99": float(N99[i, j])})
    run.flush()
    with open(os.path.join(args.csv_dir, "fact_agreement_matrix.csv"), "w") as fh:
        fh.write("model,family," + ",".join(f"spearman__{m}" for m in order) + "," + ",".join(f"null99__{m}" for m in order) + "\n")
        for i, m in enumerate(order):
            fh.write(f"{m},{fam[m]}," + ",".join(f"{S[i, j]:.6f}" for j in range(len(order))) + "," + ",".join(f"{N99[i, j]:.6f}" for j in range(len(order))) + "\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"agreement  {agreement_v}   median pairwise within-relation Spearman {med:.4f} (min {lo:.4f}, max {float(vals.max()):.4f}) over {n_pairs} pairs of {len(order)} models; "
          f"{above} of {n_pairs} pairs above the null's 99th percentile (null median {float(np.median(nulls)):.4f}, max {float(nulls.max()):.4f}); "
          f"{R.shape[1]} test rows in {len(groups_kept)} relations; same-family median {float(np.median(same)):.4f}, cross-family {float(np.median(cross)):.4f}", flush=True)
    for i, m in enumerate(order):
        others = [S[i, j] for j in range(len(order)) if j != i]
        print(f"  {m:45s} {fam[m]:12s} median with the others {statistics.median(others):.3f}  min {min(others):.3f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
