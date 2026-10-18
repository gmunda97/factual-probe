# runs the ladder for each of the 19 models with one fixed protocol: last token pooling (modernbert mean
# pooled), final norm frame, pca 768, temp 0.05, sweep every 4th layer and refine around the best
# ROSTER below is the 19 model list the other scripts import
# run: python -m experiments.main_results.all_models --model Qwen/Qwen3.5-0.8B

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from factgrasp import paths
from factgrasp import encoders as E, registry

NAME = "all_models"
RUNS = os.path.join(paths.ROOT, "results", "runs")
os.makedirs(RUNS, exist_ok=True)

THRESHOLDS = {
    "atlas_level_green_share": 1.0, "atlas_level_amber_share": 0.8, "atlas_level_red_failures": 2,
    "atlas_objective_gap_sign_test_p": 0.01, "atlas_objective_gap_amber_share": 0.8, "atlas_objective_gap_red_reversals": 2,
    "atlas_order_amber_inversions": 1, "atlas_order_red_inversions": 2,
    "atlas_post_training_tost_margin": 0.02, "atlas_post_training_green_pairs": 3, "atlas_post_training_amber_pairs": 2,
}

ROSTER = {
    "Qwen/Qwen3.5-0.8B": ("Qwen3.5", 0.873, "post"),
    "Qwen/Qwen3.5-2B": ("Qwen3.5", 2.274, "post"),
    "Qwen/Qwen3.5-4B": ("Qwen3.5", 4.660, "post"),
    "Qwen/Qwen3.5-4B-Base": ("Qwen3.5", 4.660, "base"),
    "Qwen/Qwen3.5-9B": ("Qwen3.5", 9.653, "post"),
    "Qwen/Qwen3.5-27B-GPTQ-Int4": ("Qwen3.5", 27.78, "post"),
    "google/gemma-4-E2B": ("Gemma-4", 5.123, "base"),
    "google/gemma-4-E4B": ("Gemma-4", 7.996, "base"),
    "google/gemma-4-E4B-it": ("Gemma-4", 7.996, "post"),
    "google/gemma-4-12B": ("Gemma-4", 11.960, "base"),
    "meta-llama/Llama-3.1-8B": ("Llama-3.1", 8.030, "base"),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1", 8.030, "post"),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2", 1.236, "base"),
    "meta-llama/Llama-3.2-3B": ("Llama-3.2", 3.213, "base"),
    "allenai/Olmo-3-1025-7B": ("OLMo-3", 7.298, "base"),
    "allenai/Olmo-3-7B-Instruct": ("OLMo-3", 7.298, "post"),
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": ("Ministral-3", 8.918, "post"),
    "ibm-granite/granite-4.2-8b": ("Granite-4.2", 8.792, "base"),
    "answerdotai/ModernBERT-base": ("ModernBERT", 0.150, "base"),
    "HuggingFaceTB/SmolLM3-3B": ("SmolLM3", 3.075, "post"),
}
PAIRS = [("Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B"),
         ("google/gemma-4-E4B", "google/gemma-4-E4B-it"),
         ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
         ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct")]
SIZE_LADDERS = {"Qwen3.5": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
                            "Qwen/Qwen3.5-9B"],
                "Gemma-4": ["google/gemma-4-E2B", "google/gemma-4-E4B",
                            "google/gemma-4-12B"]}
BIG_B = 12.0


ENCODERS = {"answerdotai/ModernBERT-base"}


def pooling_for(model_id: str) -> str:
    return "mean" if model_id in ENCODERS else "last"


def ladder_path(model_id: str) -> str:
    tag = E.model_slug(model_id)
    if pooling_for(model_id) != "last":
        tag = f"{tag}_{pooling_for(model_id)}"
    for d in (RUNS, paths.RUNS):
        p = os.path.join(d, f"ladder_{tag}.json")
        if os.path.exists(p):
            return p
    return os.path.join(RUNS, f"ladder_{tag}.json")


def finished(model_id: str) -> bool:
    p = ladder_path(model_id)
    if not os.path.exists(p):
        return False
    try:
        return (json.load(open(p)).get("summary") or {}).get(
            "headline_MRR_mixture_infonce") is not None
    except (json.JSONDecodeError, OSError):
        return False


def run_one(model_id: str, args) -> dict:
    size = ROSTER.get(model_id, ("?", 0.0, "?"))[1]
    cmd = [sys.executable, "-u", "-m", "experiments.main_results.ladder",
           "--model", model_id, "--seeds", args.seeds,
           "--force-pooling", pooling_for(model_id),
           "--frame", "final", "--pca", str(args.pca), "--temp", str(args.temp),
           "--out-dir-runs", "1"]
    cmd = [c for c in cmd if c != "--out-dir-runs" and c != "1"]
    if size > BIG_B:
        cmd += ["--grid-depths", args.big_depths]
    else:
        cmd += ["--grid-every", str(args.grid_every)]
    if pooling_for(model_id) != "last":
        cmd += ["--tag", f"{E.model_slug(model_id)}_{pooling_for(model_id)}"]
    if args.smoke:
        cmd += ["--smoke"]
    print(f"[{NAME}] ladder: {' '.join(cmd[3:])}", flush=True)
    rc = subprocess.run(cmd, cwd=paths.CODE).returncode
    out = {"model": model_id, "ladder_rc": rc}
    if rc != 0:
        out["error"] = f"ladder exited {rc}"
        return out
    return out


def collect() -> dict:
    rows = {}
    for model_id, (fam, size, kind) in ROSTER.items():
        p = ladder_path(model_id)
        if not os.path.exists(p):
            continue
        try:
            blob = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        s = blob.get("summary") or {}
        if s.get("headline_MRR_mixture_infonce") is None:
            continue
        gm = s.get("gap_mixture_vs_single") or {}
        gl = s.get("gap_infonce_vs_cosine") or {}
        by_row = {}
        for k, v in (blob.get("rows") or {}).items():
            if not k.startswith("{") or not isinstance(v, dict) or "MRR" not in v:
                continue
            try:
                cfg = json.loads(k)
            except Exception:
                continue
            if cfg.get("stage") != 2 or cfg.get("layer") != s.get("best_layer"):
                continue
            by_row.setdefault(v.get("row") or cfg.get("row"), []).append(float(v["MRR"]))

        def row_mrr(name):
            vs = by_row.get(name)
            return sum(vs) / len(vs) if vs else None
        rows[model_id] = {
            "model": model_id, "family": fam, "size_b": size, "base_or_post": kind,
            "layers": s.get("n_layers"), "best_layer": s.get("best_layer"),
            "relative_depth": s.get("lambda_star"), "pooling": s.get("pooling"),
            "MRR_mixture_infonce": s.get("headline_MRR_mixture_infonce"),
            "MRR_single_infonce": s.get("MRR_single_infonce"),
            "gap_infonce_minus_cosine": gl.get("diff"),
            "gap_infonce_significant": bool(gl.get("significant")),
            "gap_mixture_minus_single": gm.get("diff"),
            "gap_mixture_significant": bool(gm.get("significant")),
            "mixture_gain_under_cosine": s.get("mixture_gain_under_cosine"),
            "mixture_gain_under_infonce": s.get("mixture_gain_under_infonce"),
            "source": os.path.relpath(p, paths.ROOT),
        }
        if rows[model_id]["mixture_gain_under_cosine"] is None:
            mc, sc = row_mrr("mixture8_cosine"), row_mrr("single_cosine")
            mi, si = row_mrr("mixture8_infonce"), row_mrr("single_infonce")
            if None not in (mc, sc, mi, si):
                rows[model_id]["mixture_gain_under_cosine"] = mc - sc
                rows[model_id]["mixture_gain_under_infonce"] = mi - si
    return rows


def sign_test(n_success: int, n: int) -> float:
    if n == 0:
        return float("nan")
    def C(a, b):
        return math.comb(a, b)
    tail = sum(C(n, k) for k in range(n_success, n + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def roster_verdicts(rows: dict) -> dict:
    out, measured, verdicts = {}, {}, {}
    vals = list(rows.values())
    n = len(vals)

    sig = [r for r in vals if r["gap_infonce_significant"] and (r["gap_infonce_minus_cosine"] or 0) > 0]
    share = len(sig) / n if n else float("nan")
    out["atlas_level"] = {"n_models": n, "n_positive_significant": len(sig), "share": share,
                   "failures": [r["model"] for r in vals if r not in sig]}
    verdicts["atlas_level"] = ("green" if n and share >= THRESHOLDS["atlas_level_green_share"]
                        else "amber" if share >= THRESHOLDS["atlas_level_amber_share"]
                        else "red")
    measured["atlas_level"] = (f"InfoNCE beats cosine with the interval excluding zero on "
                        f"{len(sig)} of {n} models")

    have = [r for r in vals if r["mixture_gain_under_cosine"] is not None]
    smaller = [r for r in have
               if r["mixture_gain_under_infonce"] < r["mixture_gain_under_cosine"]]
    p = sign_test(len(smaller), len(have))
    out["atlas_objective_gap"] = {"n_models": len(have), "n_smaller_under_infonce": len(smaller),
                   "sign_test_p": p,
                   "reversals": [r["model"] for r in have if r not in smaller]}
    verdicts["atlas_objective_gap"] = ("green" if have and len(smaller) == len(have)
                        and p < THRESHOLDS["atlas_objective_gap_sign_test_p"]
                        else "amber" if have and len(smaller) / len(have) >= THRESHOLDS["atlas_objective_gap_amber_share"]
                        else "red")
    measured["atlas_objective_gap"] = (f"the mixture gain is smaller under InfoNCE than under cosine on "
                        f"{len(smaller)} of {len(have)} models, sign test p = {p:.2g}")

    lad = {}
    for fam, ids in SIZE_LADDERS.items():
        pts = [(rows[m]["size_b"], rows[m]["MRR_mixture_infonce"], m)
               for m in ids if m in rows]
        pts.sort()
        inv = sum(1 for a, b in zip(pts, pts[1:]) if b[1] < a[1] - 1e-9)
        lad[fam] = {"models": [p[2] for p in pts], "sizes": [p[0] for p in pts],
                    "MRR": [p[1] for p in pts], "inversions": inv, "n": len(pts)}
    total_inv = sum(v["inversions"] for v in lad.values())
    out["atlas_order"] = lad
    if not any(v["n"] >= 3 for v in lad.values()):
        verdicts["atlas_order"] = "not measurable yet"
    else:
        verdicts["atlas_order"] = ("green" if total_inv == 0
                            else "amber" if total_inv <= THRESHOLDS["atlas_order_amber_inversions"]
                            else "red")
    measured["atlas_order"] = (f"{total_inv} inversion(s) across "
                        + ", ".join(f"{k} {v['n']} sizes" for k, v in lad.items()))

    pr = []
    for base, post in PAIRS:
        if base in rows and post in rows:
            d = rows[post]["MRR_mixture_infonce"] - rows[base]["MRR_mixture_infonce"]
            pr.append({"base": base, "post": post, "diff": d,
                       "within_margin": abs(d) <= THRESHOLDS["atlas_post_training_tost_margin"],
                       "MRR_base": rows[base]["MRR_mixture_infonce"],
                       "MRR_post": rows[post]["MRR_mixture_infonce"]})
    ok = sum(1 for x in pr if x["within_margin"])
    out["atlas_post_training"] = {"pairs": pr, "n_pairs": len(pr), "n_within_margin": ok}
    if not pr:
        verdicts["atlas_post_training"] = "not measurable yet"
    else:
        verdicts["atlas_post_training"] = ("green" if ok >= THRESHOLDS["atlas_post_training_green_pairs"]
                            else "amber" if ok >= THRESHOLDS["atlas_post_training_amber_pairs"] else "red")
    measured["atlas_post_training"] = (f"{ok} of {len(pr)} base and post-trained pairs differ by at most "
                        f"{THRESHOLDS['atlas_post_training_tost_margin']} MRR")
    return {"detail": out, "verdicts": verdicts, "measured": measured}


def main() -> int:
    ap = argparse.ArgumentParser(description="run the ladder for the 19 models with the fixed protocol")
    ap.add_argument("--model", default="")
    ap.add_argument("--roster-only", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--grid-every", type=int, default=4)
    ap.add_argument("--big-depths", default="0.5,0.8,1.0")
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    if args.roster_only:
        rows = collect()
        run = registry.Run(NAME, "roster", thresholds=THRESHOLDS, out_dir=RUNS)
        skipped_path = os.path.join(paths.LOGS, "atlas_skipped.json")
        skipped = json.load(open(skipped_path)) if os.path.exists(skipped_path) else {}
        res = roster_verdicts(rows) if rows else {"detail": {}, "verdicts": {},
                                                  "measured": {}}
        summary = {"n_models": len(rows), "models": sorted(rows),
                   "skipped": skipped, "per_model": rows, **res}
        run.set_summary(summary)
        judged = [v for v in res["verdicts"].values() if v in ("red", "amber", "green")]
        run.set_verdict(min(judged, key=lambda x: ["red", "amber", "green"].index(x))
                        if judged else "n/a")
        run.note("A model that failed to load twice is listed in `skipped` with its error; "
                 "no model is dropped silently.")
        srows = [{"hypothesis": k, "verdict": v, "measured": res["measured"].get(k, "")[:70]}
                 for k, v in sorted(res["verdicts"].items())]
        srows.append({"hypothesis": "models", "verdict": len(rows),
                      "measured": ", ".join(sorted(r.split("/")[-1] for r in rows))[:70]})
        run.summary(srows)
        return 0

    if not args.model:
        raise SystemExit(f"[{NAME}] pass --model or --roster-only")
    if finished(args.model):
        print(f"[{NAME}] {args.model}: ladder already finished, skipping", flush=True)
        return 0

    last = {}
    for attempt in range(1, args.retries + 1):
        print(f"[{NAME}] {args.model}: attempt {attempt} of {args.retries}", flush=True)
        last = run_one(args.model, args)
        if last.get("ladder_rc") == 0:
            return 0
        if last.get("ladder_rc") == 4:
            print(f"[{NAME}] yielded the GPU; rerun to continue", flush=True)
            return 4
    path = os.path.join(paths.LOGS, "atlas_skipped.json")
    skipped = json.load(open(path)) if os.path.exists(path) else {}
    skipped[args.model] = {"error": last.get("error", "unknown"),
                           "attempts": args.retries}
    with open(path, "w") as fh:
        json.dump(skipped, fh, indent=2)
    print(f"[{NAME}] {args.model} failed {args.retries} times; recorded in {path} and "
          f"skipped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
