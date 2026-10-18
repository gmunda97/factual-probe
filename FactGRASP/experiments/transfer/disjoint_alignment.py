# the 18761 pair result of sec 4.2: refit every alignment only on training pairs whose subject and
# object strings never appear in test (90.7 percent) and on a random subset of the same size (95.8 percent)

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

from factgrasp import crossmodel as X, data as D, paths, registry
from experiments.transfer import transfer_all_pairs as TP

NAME = "disjoint_alignment"
RUNS = os.path.join(paths.ROOT, "results", "runs")
PAIRS_PATH = os.path.join(paths.ROOT, "results", "runs", "transfer_all_pairs_all.json")
THRESHOLDS = {
    "disjoint_alignment": "string identity does not carry the transfer: median efficiency over the 342 pairs within 0.05 of the "
           "shared-string median, and at least 90 percent of pairs within 0.10 (green); median within 0.15 (amber); "
           "more (red)",
}


def _norm(s) -> str:
    return str(s).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--row-control", action="store_true", help="add the same-size random-subset stitch (extra column)")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    device = torch.device("cpu")
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    pairs = [v for v in json.load(open(PAIRS_PATH))["rows"].values() if isinstance(v, dict) and "tau" in v]
    stored = {(r["A"], r["B"]): r for r in pairs}
    models = sorted({r["A"] for r in pairs})
    te, tr = splits["test"], splits["train"]
    test_strings = {_norm(x) for x in te["subject"]} | {_norm(x) for x in te["object"]}
    keep = torch.tensor([not (_norm(s) in test_strings or _norm(o) in test_strings) for s, o in zip(tr["subject"], tr["object"])])
    n_keep = int(keep.sum())
    print(f"[{NAME}] {n_keep} of {keep.numel()} training rows have a subject and an object absent from every test triple", flush=True)
    g = torch.Generator().manual_seed(0)
    rand_idx = torch.randperm(keep.numel(), generator=g)[:n_keep].sort().values
    spaces = {}
    t0 = time.time()
    for m in models:
        spaces[m] = TP.ModelSpace(m, splits, device, SimpleNamespace(temp=0.05, epochs=60))
        print(f"[{NAME}] loaded {m} (native {spaces[m].mrr_native:.4f}); {time.time() - t0:.0f}s", flush=True)
    rows = []
    for A in models:
        a = spaces[A]
        for B in models:
            if B == A:
                continue
            key = registry.config_key({"A": A, "B": B, "ridge": args.ridge})
            if run.done(key):
                row = run.get(key)
                if args.row_control and "tau_random_subset" not in row:
                    b = spaces[B]
                    S_r = X.fit_stitch(b.x_train_cpu[rand_idx], a.x_train_cpu[rand_idx], ridge=args.ridge)
                    mrr_r, _ = a.stitched(b, S_r, splits)
                    row["mrr_stitched_random_subset"] = mrr_r; row["tau_random_subset"] = X.transfer_efficiency(mrr_r, a.mrr_native)
                    run.put(key, row)
                rows.append(row); continue
            b = spaces[B]
            S_full = X.fit_stitch(b.x_train_cpu, a.x_train_cpu, ridge=args.ridge)
            mrr_full, _ = a.stitched(b, S_full, splits)
            S_dis = X.fit_stitch(b.x_train_cpu[keep], a.x_train_cpu[keep], ridge=args.ridge)
            mrr_dis, _ = a.stitched(b, S_dis, splits)
            row = {"A": A, "B": B, "family_A": stored[(A, B)]["family_A"], "family_B": stored[(A, B)]["family_B"],
                   "kind_A": stored[(A, B)]["kind_A"], "kind_B": stored[(A, B)]["kind_B"],
                   "n_fit_rows": n_keep, "mrr_native": a.mrr_native, "mrr_stitched_shared": mrr_full,
                   "tau_shared": X.transfer_efficiency(mrr_full, a.mrr_native), "tau_shared_stored_transfer_all_pairs": stored[(A, B)]["tau"],
                   "mrr_stitched_disjoint": mrr_dis, "tau_disjoint": X.transfer_efficiency(mrr_dis, a.mrr_native)}
            row["delta"] = row["tau_disjoint"] - row["tau_shared"]
            if args.row_control:
                S_r = X.fit_stitch(b.x_train_cpu[rand_idx], a.x_train_cpu[rand_idx], ridge=args.ridge)
                mrr_r, _ = a.stitched(b, S_r, splits)
                row["mrr_stitched_random_subset"] = mrr_r; row["tau_random_subset"] = X.transfer_efficiency(mrr_r, a.mrr_native)
            run.put(key, row); rows.append(row)
        run.flush()
        print(f"[{NAME}] target {A}: done ({time.time() - t0:.0f}s)", flush=True)
    taus_s = [r["tau_shared"] for r in rows]; taus_d = [r["tau_disjoint"] for r in rows]; deltas = [r["delta"] for r in rows]
    med_s, med_d = statistics.median(taus_s), statistics.median(taus_d)
    within10 = sum(1 for d in deltas if abs(d) <= 0.10) / len(deltas)
    gap = abs(med_d - med_s)
    verdict = "green" if (gap <= 0.05 and within10 >= 0.9) else "amber" if gap <= 0.15 else "red"
    run.set_summary({"verdicts": {"disjoint_alignment": verdict},
                     "measured": {"disjoint_alignment_median_tau_shared": round(med_s, 4), "disjoint_alignment_median_tau_disjoint": round(med_d, 4), "disjoint_alignment_median_gap": round(gap, 4),
                                  "disjoint_alignment_share_within_0.10": round(within10, 4), "disjoint_alignment_pairs": len(rows), "disjoint_alignment_n_fit_rows": n_keep,
                                  "disjoint_alignment_delta_median": round(statistics.median(deltas), 4), "disjoint_alignment_delta_min": round(min(deltas), 4), "disjoint_alignment_delta_max": round(max(deltas), 4),
                                  "disjoint_alignment_delta_quartiles": [round(q, 4) for q in statistics.quantiles(deltas, n=4)],
                                  "disjoint_alignment_pairs_outside_0.10": [f"{r['A']} <- {r['B']} ({r['delta']:+.3f})" for r in rows if abs(r["delta"]) > 0.10],
                                  "disjoint_alignment_decoder_decoder_delta_median": round(statistics.median([r["delta"] for r in rows if r["kind_A"] == r["kind_B"] == "decoder"]), 4),
                                  "disjoint_alignment_encoder_involved_delta_median": round(statistics.median([r["delta"] for r in rows if "encoder" in (r["kind_A"], r["kind_B"])]), 4),
                                  "disjoint_alignment_row_control_median_tau_random_subset": (round(statistics.median([r["tau_random_subset"] for r in rows]), 4) if all("tau_random_subset" in r for r in rows) else None),
                                  "disjoint_alignment_row_control_median_disjoint_minus_random": (round(statistics.median([r["tau_disjoint"] - r["tau_random_subset"] for r in rows]), 4) if all("tau_random_subset" in r for r in rows) else None),
                                  "disjoint_alignment_row_control_share_disjoint_within_0.05_of_random": (round(sum(1 for r in rows if abs(r["tau_disjoint"] - r["tau_random_subset"]) <= 0.05) / len(rows), 4) if all("tau_random_subset" in r for r in rows) else None)}})
    run.flush()
    with open(os.path.join(args.csv_dir, "disjoint_alignment.csv"), "w") as fh:
        fh.write("pair,A,B,family_A,family_B,n_fit_rows,mrr_native,mrr_stitched_shared,tau_shared,tau_shared_stored_transfer_all_pairs,mrr_stitched_disjoint,tau_disjoint,delta,"
                 "tau_random_subset,disjoint_minus_random_subset\n")
        for r in rows:
            tr_ = "" if "tau_random_subset" not in r else f"{r['tau_random_subset']:.6f}"
            dr_ = "" if "tau_random_subset" not in r else f"{r['tau_disjoint'] - r['tau_random_subset']:+.6f}"
            fh.write(f"{r['A']} <- {r['B']},{r['A']},{r['B']},{r['family_A']},{r['family_B']},{r['n_fit_rows']},{r['mrr_native']:.6f},{r['mrr_stitched_shared']:.6f},"
                     f"{r['tau_shared']:.6f},{r['tau_shared_stored_transfer_all_pairs']:.6f},{r['mrr_stitched_disjoint']:.6f},{r['tau_disjoint']:.6f},{r['delta']:+.6f},{tr_},{dr_}\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"disjoint_alignment  {verdict}   median tau shared {med_s:.4f}, disjoint {med_d:.4f} (gap {gap:.4f}); {100 * within10:.1f} percent of {len(rows)} pairs within 0.10 "
          f"(delta median {statistics.median(deltas):+.4f}, range {min(deltas):+.4f} to {max(deltas):+.4f}); {n_keep} fitting rows per pair", flush=True)
    if all("tau_random_subset" in r for r in rows):
        dr = [r["tau_disjoint"] - r["tau_random_subset"] for r in rows]
        print(f"row control (extra): median tau on a same-size random subset {statistics.median(r['tau_random_subset'] for r in rows):.4f}; disjoint minus random-subset "
              f"median {statistics.median(dr):+.4f}, range {min(dr):+.4f} to {max(dr):+.4f}; {100 * sum(1 for d in dr if abs(d) <= 0.05) / len(dr):.1f} percent within 0.05", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
