# controls on the stitching result, the mismatched pairs (derangement) giving the 0.007 median
# and the reconstruction r2 of each alignment

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from types import SimpleNamespace

import torch

from factgrasp import crossmodel as X, data as D, paths, registry
from experiments.transfer import transfer_all_pairs as TP

NAME = "transfer_controls"
RUNS = os.path.join(paths.ROOT, "results", "runs")
PAIRS_PATH = os.path.join(RUNS, "transfer_all_pairs_all.json")
FRESH_PATH = os.path.join(RUNS, "fresh_map_comparison_all.json")
THRESHOLDS = {
    "reconstruction": "R2 of the stitch on held-out test queries, per ordered pair; reported, no threshold",
    "fresh_map": "portability = MRR(stitched) / MRR(fresh probe, 3 seeds) on 8 pairs: at least 0.9 "
           "on 7 of 8 (green); 0.75 to 0.9 (amber); below 0.75 on 3 or more (red, and the "
           "paper says a retrained probe is needed, not a transferred one)",
    "direction": "over unordered decoder pairs, sign(tau(A<-B) - tau(B<-A)) = sign(MRR_B - MRR_A) on "
           "at least 80 percent, and tau(A<-B) above 1 on at least 80 percent of ordered pairs "
           "where B's native MRR exceeds A's by 0.05 or more (green); 65 to 80 percent (amber); "
           "below 65 percent (red). The band is set by the weaker of the two percentages, "
           "fixed before either was computed",
    "post_trained_source": "median tau with a post-trained Gemma source below the overall median by at least "
           "0.15, and no other family's sources below the overall median by more than 0.05 "
           "(green); one clause (amber); neither (red). Groups are source family by training "
           "stage, medians pooled over the group's ordered pairs; every group other than the "
           "post-trained Gemma one counts for the second clause, the encoder included, as written",
}


def pair_rows() -> list:
    d = json.load(open(PAIRS_PATH))
    return [v for v in d["rows"].values() if isinstance(v, dict) and "tau" in v]


def stage_of(roster, m):
    return roster[m][2] if m in roster else "base"


def part_e(run, ridge: float) -> None:
    rows = pair_rows()
    names = sorted({r["A"] for r in rows} | {r["B"] for r in rows})
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    args = SimpleNamespace(temp=0.05, epochs=60)
    device = torch.device("cpu")
    spaces = {}
    for m in names:
        sp = TP.ModelSpace(m, splits, device, args)
        h_o = sp.pool_cpu[sp.gold]
        cos = torch.nn.functional.cosine_similarity(sp.u.float(), h_o.float(), dim=-1).clamp(-1, 1)
        ang = torch.acos(cos)
        key = registry.config_key({"model": m, "part": "margin"})
        run.put(key, {"part": "margin", "model": m, "layer": sp.layer,
                      "mrr_native": sp.mrr_native,
                      "mean_margin_deg": float(sp.gamma.mean()) * 180 / math.pi,
                      "median_margin_deg": float(sp.gamma.median()) * 180 / math.pi,
                      "mean_half_margin_deg": float(sp.gamma.mean()) * 90 / math.pi,
                      "mean_native_angle_deg": float(ang.mean()) * 180 / math.pi,
                      "share_native_angle_under_half_margin":
                          float((ang < sp.gamma.float() / 2).float().mean())})
        spaces[m] = SimpleNamespace(x_train=sp.x_train_cpu, x_test=sp.x_test.cpu())
        del sp
        print(f"[{NAME}] {m}: queries loaded", flush=True)
    for A in names:
        a = spaces[A]
        ss_tot = float(((a.x_test - a.x_test.mean(0, keepdim=True)) ** 2).sum())
        ss_tot_tr = float(((a.x_train - a.x_train.mean(0, keepdim=True)) ** 2).sum())
        for B in names:
            if A == B:
                continue
            key = registry.config_key({"A": A, "B": B, "part": "e"})
            if run.done(key):
                continue
            b = spaces[B]
            S = X.fit_stitch(b.x_train, a.x_train, ridge=ridge)
            r2 = 1.0 - float(((a.x_test - b.x_test @ S) ** 2).sum()) / ss_tot
            r2_tr = 1.0 - float(((a.x_train - b.x_train @ S) ** 2).sum()) / ss_tot_tr
            run.data["rows"][key] = {"part": "e", "A": A, "B": B, "r2_test": r2, "r2_train": r2_tr}
        run.flush()
        print(f"[{NAME}] {A} <- all: R2 done", flush=True)


def part_g(run) -> dict:
    rows = pair_rows()
    dec = [r for r in rows if r["kind_A"] == r["kind_B"] == "decoder"]
    by = {(r["A"], r["B"]): r for r in dec}
    mrr = {}
    for r in dec:
        mrr[r["A"]] = r["mrr_native"]
    seen, agree, n_unordered, ties = set(), 0, 0, 0
    for (A, B), r in by.items():
        if (B, A) not in by or frozenset((A, B)) in seen:
            continue
        seen.add(frozenset((A, B)))
        asym = r["tau"] - by[(B, A)]["tau"]
        dm = mrr[B] - mrr[A]
        if asym == 0 or dm == 0:
            ties += 1
        n_unordered += 1
        if (asym > 0) == (dm > 0):
            agree += 1
    strong = [r for r in dec if mrr[r["B"]] - mrr[r["A"]] >= 0.05]
    above = sum(1 for r in strong if r["tau"] > 1.0)
    share_agree = agree / n_unordered if n_unordered else float("nan")
    share_above = above / len(strong) if strong else float("nan")
    weaker = min(share_agree, share_above)
    verdict = "green" if weaker >= 0.80 else "amber" if weaker >= 0.65 else "red"
    out = {"part": "g", "n_unordered_decoder_pairs": n_unordered, "sign_agreements": agree,
           "sign_ties": ties, "share_sign_agreement": share_agree,
           "n_ordered_source_ahead_by_0.05": len(strong), "tau_above_1_among_them": above,
           "share_tau_above_1": share_above, "weaker_share": weaker, "verdict": verdict}
    run.put(registry.config_key({"part": "g"}), out)
    return out


def part_h(run) -> dict:
    rows = pair_rows()
    roster = TP._roster()
    overall = statistics.median(r["tau"] for r in rows)
    groups = {}
    per_source = {}
    for r in rows:
        g = (r["family_B"], stage_of(roster, r["B"]))
        groups.setdefault(g, []).append(r["tau"])
        per_source.setdefault(r["B"], []).append(r["tau"])
    med = {f"{f} / {s}": statistics.median(v) for (f, s), v in groups.items()}
    n = {f"{f} / {s}": len(v) for (f, s), v in groups.items()}
    post_gemma = [g for g in med if g.startswith("Gemma-4 /") and g.endswith("post")]
    pg = med[post_gemma[0]] if post_gemma else float("nan")
    clause1 = bool(post_gemma) and (overall - pg) >= 0.15
    others = {g: overall - m for g, m in med.items() if g not in post_gemma}
    worst = max(others.items(), key=lambda kv: kv[1]) if others else ("", float("nan"))
    clause2 = bool(others) and all(d <= 0.05 for d in others.values())
    dec_only = {g: d for g, d in others.items() if not g.startswith("ModernBERT")}
    clause2_decoders = bool(dec_only) and all(d <= 0.05 for d in dec_only.values())
    verdict = "green" if (clause1 and clause2) else "amber" if (clause1 or clause2) else "red"
    sources = {m: {"family": roster[m][0] if m in roster else "", "stage": stage_of(roster, m),
                   "median_tau_as_source": statistics.median(v), "n": len(v)}
               for m, v in per_source.items()}
    out = {"part": "h", "overall_median_tau": overall, "group_median_tau": med, "group_n": n,
           "post_trained_gemma_group": post_gemma, "post_trained_gemma_median": pg,
           "clause1_post_gemma_below_by_0.15": clause1,
           "clause2_no_other_group_below_by_0.05": clause2,
           "clause2_worst_other_group": {"group": worst[0], "below_overall_by": worst[1]},
           "clause2_over_decoder_groups_only_reported": clause2_decoders,
           "verdict": verdict, "sources": sources}
    run.put(registry.config_key({"part": "h"}), out)
    return out


def collect(run, csv_dir: str) -> int:
    rows = run.rows()
    g = rows.get(registry.config_key({"part": "g"})) or part_g(run)
    h = rows.get(registry.config_key({"part": "h"})) or part_h(run)
    e_rows = [v for v in rows.values() if isinstance(v, dict) and v.get("part") == "e"]
    margins = [v for v in rows.values() if isinstance(v, dict) and v.get("part") == "margin"]
    f = None
    if os.path.exists(FRESH_PATH):
        fd = json.load(open(FRESH_PATH))
        f = fd.get("summary") or None
    verd = {"reconstruction": "reported" if e_rows else "pending",
            "fresh_map": (f or {}).get("verdict", "pending"),
            "direction": g["verdict"], "post_trained_source": h["verdict"]}
    r2s = [r["r2_test"] for r in e_rows]
    measured = {
        "reconstruction_n_pairs": len(e_rows),
        "reconstruction_median_r2_test": (round(statistics.median(r2s), 4) if r2s else None),
        "reconstruction_min_r2_test": (round(min(r2s), 4) if r2s else None),
        "reconstruction_max_r2_test": (round(max(r2s), 4) if r2s else None),
        "reconstruction_spearman_r2_tau": None,
        "certificate_mean_half_margin_deg": (round(statistics.mean(m["mean_half_margin_deg"]
                                                                   for m in margins), 2)
                                             if margins else None),
        "certificate_mean_native_angle_deg": (round(statistics.mean(m["mean_native_angle_deg"]
                                                                    for m in margins), 2)
                                              if margins else None),
        "fresh_map_portability": (f or {}).get("portability"),
        "fresh_map_pairs_at_least_0.9": (f or {}).get("n_at_least_0.9"),
        "fresh_map_pairs_below_0.75": (f or {}).get("n_below_0.75"),
        "direction_share_sign_agreement": round(g["share_sign_agreement"], 4),
        "direction_share_tau_above_1_source_ahead": round(g["share_tau_above_1"], 4),
        "direction_n_unordered": g["n_unordered_decoder_pairs"],
        "direction_n_source_ahead": g["n_ordered_source_ahead_by_0.05"],
        "post_trained_source_overall_median": round(h["overall_median_tau"], 4),
        "post_trained_source_post_gemma_median": (round(h["post_trained_gemma_median"], 4)
                                  if h["post_trained_gemma_group"] else None),
        "post_trained_source_worst_other_group": h["clause2_worst_other_group"],
        "post_trained_source_clause2_decoders_only_reported": h["clause2_over_decoder_groups_only_reported"],
        "post_trained_source_group_median_tau": {k: round(v, 4) for k, v in h["group_median_tau"].items()},
    }
    if e_rows:
        from factgrasp import stats
        pairs = {(r["A"], r["B"]): r["tau"] for r in pair_rows()}
        paired = [(r["r2_test"], pairs[(r["A"], r["B"])]) for r in e_rows if (r["A"], r["B"]) in pairs]
        measured["reconstruction_spearman_r2_tau"] = round(stats.spearman([p[0] for p in paired],
                                                               [p[1] for p in paired]), 4)
    run.set_summary({"verdicts": verd, "measured": measured})
    run.flush()
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, "transfer_sources.csv"), "w") as fh:
        fh.write("source,family,stage,n_pairs_as_source,median_tau_as_source,"
                 "mean_half_margin_deg,mean_native_angle_deg,mrr_native\n")
        mg = {m["model"]: m for m in margins}
        for m, v in sorted(h["sources"].items()):
            mm = mg.get(m, {})
            fh.write(f"{m},{v['family']},{v['stage']},{v['n']},{v['median_tau_as_source']:.6f},"
                     f"{mm.get('mean_half_margin_deg', float('nan')):.4f},"
                     f"{mm.get('mean_native_angle_deg', float('nan')):.4f},"
                     f"{mm.get('mrr_native', float('nan')):.6f}\n")
    TP.collect(csv_dir)
    print(f"=== SUMMARY {NAME} ===", flush=True)
    for k, v in verd.items():
        print(f"{k}  {v}", flush=True)
    for k in ("reconstruction_median_r2_test", "reconstruction_min_r2_test", "direction_share_sign_agreement",
              "direction_share_tau_above_1_source_ahead", "post_trained_source_post_gemma_median",
              "post_trained_source_worst_other_group", "certificate_mean_half_margin_deg",
              "certificate_mean_native_angle_deg", "fresh_map_portability"):
        print(f"{k}: {measured[k]}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="egh", help="any of e, g, h; default all three")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, "all", config={"ridge": args.ridge, "part": args.part},
                       thresholds=THRESHOLDS, out_dir=RUNS)
    if args.collect:
        return collect(run, args.csv_dir)
    if "e" in args.part:
        part_e(run, args.ridge)
    if "g" in args.part:
        g = part_g(run)
        print(f"[{NAME}] direction {g['verdict']}: sign agreement {g['share_sign_agreement']:.3f} on "
              f"{g['n_unordered_decoder_pairs']} unordered decoder pairs; tau above 1 on "
              f"{g['share_tau_above_1']:.3f} of {g['n_ordered_source_ahead_by_0.05']} "
              f"source-ahead pairs", flush=True)
    if "h" in args.part:
        h = part_h(run)
        print(f"[{NAME}] post_trained_source {h['verdict']}: overall median {h['overall_median_tau']:.3f}; "
              f"post-trained Gemma sources {h['post_trained_gemma_median']:.3f}; worst other "
              f"group {h['clause2_worst_other_group']}", flush=True)
    return collect(run, args.csv_dir)


if __name__ == "__main__":
    sys.exit(main())
