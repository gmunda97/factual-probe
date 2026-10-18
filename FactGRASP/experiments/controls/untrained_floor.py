# untrained model floor: same architecture with random weights, same layer pooling pca and map
# gives the floor band in figure 4a and the twin floor in the fact agreement controls
# run: python -m experiments.controls.untrained_floor --models meta-llama/Llama-3.1-8B

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

from factgrasp import (data as D, encoders as E, gpu_guard, knowledge as KN, layers as L,
                  metrics as M, paths, registry, stats)
from factgrasp.ladder_io import _ladder

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "untrained_floor"
RUNS = os.path.join(paths.ROOT, "results", "runs")
FIELDS = ("subject", "pred_value", "object")
ROSTER = ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
          "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-9B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct",
          "answerdotai/ModernBERT-base", "google/gemma-4-12B", "google/gemma-4-E2B", "google/gemma-4-E4B",
          "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-8b", "meta-llama/Llama-3.1-8B",
          "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
          "mistralai/Ministral-3-8B-Instruct-2512-BF16"]
THREE_SEEDS = {"Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B", "google/gemma-4-E4B"}
AT_FLOOR = {"answerdotai/ModernBERT-base": "at floor"}
THRESHOLDS = {
    "floor_gap": "the trained model is far above its untrained twin on every model: trained minus untrained "
           "MRR at least 0.15 on 17 of 19 (green); 12 to 16 (amber); fewer (red)",
    "floor_ordering": "the ordering of the atlas survives the floor: Spearman between trained MRR and (trained "
           "minus untrained) MRR across the 19 at least 0.8 (green); 0.6 to 0.8 (amber); below 0.6 "
           "(red, the atlas is reported floor-corrected only)",
}


def random_init(model_id: str, seed: int, device):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    repo, revision = E.split_revision(model_id)
    rk = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(repo, **rk)
    cfg = AutoConfig.from_pretrained(repo, **rk)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        try:
            model = AutoModelForCausalLM.from_config(cfg)
        except (ValueError, TypeError):
            model = AutoModel.from_config(cfg)
    finally:
        torch.set_default_dtype(old)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def trained_numbers(model_id: str, ladder_suffix: str = "") -> dict:
    blob = _ladder(E.model_slug(model_id) + ladder_suffix)
    s = blob.get("summary") or {}
    best = s.get("best_layer")
    rows = []
    for k, v in (blob.get("rows") or {}).items():
        try:
            cfg = json.loads(k)
        except Exception:
            continue
        if cfg.get("stage") == 2 and cfg.get("layer") == best and cfg.get("row") == "single_infonce" and "seed" in cfg:
            rows.append(v)
    if not rows:
        return {}
    N = int(rows[0].get("pool", 0))
    losses = [-math.log(r["exp_neg_L"]) for r in rows if r.get("exp_neg_L")]
    return {"layer": int(best), "pooling": s.get("pooling") or "last", "n_seeds": len(rows),
            "MRR": statistics.mean(r["MRR"] for r in rows), "pool": N,
            "bits": KN.recoverable_bits(statistics.mean(losses), N) if losses else None,
            "mixture_MRR": s.get("headline_MRR_mixture_infonce")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--ladder-suffix", default="", help="read the trained side from ladder_<slug><suffix>.json "
                    "(the atlas's mean-pooled ModernBERT row is the `_mean` ladder); a run with a suffix uses its own "
                    "tag and writes no extract")
    ap.add_argument("--smoke", action="store_true", help="one small model, 384 rows, 2 epochs")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    if args.smoke and args.tag == "all":
        args.tag = "smoke"
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    models = ROSTER if args.models == "all" else [m.strip() for m in args.models.split(",")]
    if args.smoke:
        models = ["Qwen/Qwen3.5-0.8B"]
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    if args.smoke:
        splits = {s: df.head(384).reset_index(drop=True) for s, df in splits.items()}
    import pandas as pd
    valid_map = D.valid_sets(pd.concat(list(splits.values()), ignore_index=True))
    for model_id in models:
        key = registry.config_key({"model": model_id, "smoke": args.smoke, **({"ladder_suffix": args.ladder_suffix} if args.ladder_suffix else {})})
        if run.done(key):
            continue
        tr = trained_numbers(model_id, args.ladder_suffix)
        if not tr:
            print(f"[{NAME}] {model_id}: no ladder rows; skipped", flush=True); continue
        layer, pooling = tr["layer"], tr["pooling"]
        t0 = time.time()
        rid = model_id + "-randinit"
        tags = {s: (s if not args.smoke else f"{s}n{len(splits[s])}") for s in splits}
        need = any(not os.path.exists(E.cache_path(rid, "wikidata", tags[s], f, pooling, layer))
                   for s in splits for f in FIELDS)
        tok, model = (random_init(model_id, 0, device) if need else (None, None))
        emb = {s: {f: E.load_or_embed(splits[s][f].astype(str).tolist(), tok, model, [layer], pooling,
                                      model_id=rid, dataset="wikidata", split=tags[s], field=f,
                                      batch=args.embed_batch, device=device, progress=False)[layer].float()
                   for f in FIELDS} for s in splits}
        if model is not None:
            del model; torch.cuda.empty_cache()
        cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
        proj = E.PCAProjector.fit(cat, k=args.pca, seed=0) if cat.shape[1] > args.pca else None
        P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
        Tn = {s: {f: P(emb[s][f]) for f in FIELDS} for s in emb}
        x = {s: Tn[s]["subject"] + Tn[s]["pred_value"] for s in Tn}
        names, pool, gold = C.build_unique_object_pool(splits["test"], Tn["test"]["object"])
        vlists = D.valid_index_lists(splits["test"], names, valid_map)
        seeds = [0, 1, 2] if (model_id in THREE_SEEDS and not args.smoke) else [0]
        per_seed, rr_seeds = [], []
        for seed in seeds:
            m = L.train_single_map(x["train"], Tn["train"]["pred_value"], Tn["train"]["object"], x["valid"],
                                   Tn["valid"]["pred_value"], Tn["valid"]["object"], device, loss="infonce",
                                   epochs=2 if args.smoke else args.epochs, temp=args.temp, seed=seed)
            o_hat, _ = L.predict(m, x["test"], Tn["test"]["pred_value"], device)
            res = M.evaluate(o_hat, pool, gold, valid=vlists, tie="random", tau=args.temp, seed=seed)
            loss = -math.log(res["exp_neg_L"]) if res.get("exp_neg_L") else float("inf")
            per_seed.append({**M.jsonable(res), "seed": seed, "pool_loss_nats": loss,
                             "bits": KN.recoverable_bits(loss, int(pool.shape[0]))})
            rr_seeds.append(res["rr"].float())
            del m
        u_mrr = statistics.mean(p["MRR"] for p in per_seed)
        u_bits = statistics.mean(p["bits"] for p in per_seed)
        row = {"model": model_id, "layer": layer, "pooling": pooling, "width": int(emb["train"]["subject"].shape[1]),
               "untrained_MRR": u_mrr, "untrained_bits": u_bits, "untrained_pool": int(pool.shape[0]),
               "untrained_seeds": len(seeds), "per_seed": per_seed,
               "trained_MRR": tr["MRR"], "trained_bits": tr["bits"], "trained_pool": tr["pool"],
               "trained_seeds": tr["n_seeds"], "trained_mixture_MRR": tr["mixture_MRR"],
               "delta_MRR": tr["MRR"] - u_mrr,
               "bits_added_over_untrained": (tr["bits"] - u_bits) if tr["bits"] is not None else None,
               "seconds": round(time.time() - t0)}
        run.put(key, row)
        torch.save(torch.stack(rr_seeds).mean(0), run.aux_path(f"rr_{E.model_slug(model_id)}.pt"))
        print(f"[{NAME}] {model_id}: layer {layer}, untrained MRR {u_mrr:.4f} ({u_bits:.2f} bits) against "
              f"trained {tr['MRR']:.4f} ({tr['bits']:.2f} bits); delta {row['delta_MRR']:+.4f}, bits added "
              f"{row['bits_added_over_untrained']:+.2f}; {time.time() - t0:.0f}s", flush=True)
        del emb, Tn, x
        if gpu_guard.check_yield():
            gpu_guard.yield_and_exit(f"{NAME}: after {model_id}")
    if args.smoke:
        print(f"[{NAME}] smoke run; no verdicts", flush=True)
        return 0
    if args.ladder_suffix or args.tag != "all":
        run.flush()
        print(f"[{NAME}] tag {args.tag}, ladder suffix {args.ladder_suffix!r}: rows written, no verdicts and no extract", flush=True)
        return 0
    return collect(run, args.csv_dir)


FIVE_SEED_CSV = os.path.join(paths.ROOT, "final_results", "data", "seed_counts.csv")


def reconcile(rows: list) -> tuple:
    import csv
    five = {r["model"]: r for r in csv.DictReader(open(FIVE_SEED_CSV))} if os.path.exists(FIVE_SEED_CSV) else {}
    p_mean = os.path.join(RUNS, f"{NAME}_modernbert_mean.json")
    mean_rows = [v for v in (json.load(open(p_mean)).get("rows") or {}).values()
                 if isinstance(v, dict) and "untrained_MRR" in v] if os.path.exists(p_mean) else []
    out, superseded = [], []
    for r in rows:
        r = dict(r)
        mean_r = next((mr for mr in mean_rows if mr["model"] == r["model"]), None)
        if mean_r is not None:
            superseded.append({**r, "trained_MRR_3seed": r["trained_MRR"], "trained_MRR_source": "ladder three seeds last token",
                               "status": AT_FLOOR.get(r["model"], "readout") + "; last-token convention superseded by the mean-pooled row"})
            r = dict(mean_r); r["trained_MRR_3seed"] = r["trained_MRR"]
            r["trained_MRR_source"] = "mean-pooled ladder three seeds (the atlas row; no five-seed mean-pooled row exists)"
        elif r["model"] in five:
            r["trained_MRR_3seed"] = r["trained_MRR"]
            r["trained_MRR"] = float(five[r["model"]]["MRR_single_infonce_5seed"])
            r["trained_MRR_source"] = "five-seed atlas row (seed_counts.csv)"
            if five[r["model"]]["best_layer_5seed"] != str(r["layer"]):
                r["trained_MRR_source"] += f"; five-seed layer {five[r['model']]['best_layer_5seed']} against the floor's {r['layer']}"
        else:
            r["trained_MRR_3seed"] = r["trained_MRR"]; r["trained_MRR_source"] = "ladder three seeds"
        r["delta_MRR"] = r["trained_MRR"] - r["untrained_MRR"]
        r["status"] = r.get("status") or AT_FLOOR.get(r["model"], "readout")
        out.append(r)
    return out, superseded


def collect(run, csv_dir: str) -> int:
    rows = [v for v in run.rows().values() if isinstance(v, dict) and "untrained_MRR" in v]
    rows, superseded = reconcile(rows)
    n_ok = sum(1 for r in rows if r["delta_MRR"] >= 0.15)
    tr = [r["trained_MRR"] for r in rows]; dl = [r["delta_MRR"] for r in rows]
    rho = stats.spearman(tr, dl) if len(rows) >= 3 else float("nan")
    pending = len(rows) < 19
    v_a = "pending" if pending else ("green" if n_ok >= 17 else "amber" if n_ok >= 12 else "red")
    v_b = "pending" if pending else ("green" if rho >= 0.8 else "amber" if rho >= 0.6 else "red")
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, "untrained_floor.csv"), "w") as fh:
        fh.write("model,layer,pooling,width,trained_MRR,untrained_MRR,delta_MRR,trained_bits,untrained_bits,"
                 "bits_added_over_untrained,trained_mixture_MRR,untrained_pool,trained_pool,untrained_seeds,status,"
                 "trained_MRR_3seed,trained_MRR_source\n")
        for r in sorted(rows, key=lambda r: -r["trained_MRR"]) + superseded:
            fh.write(f"{r['model']},{r['layer']},{r['pooling']},{r['width']},{r['trained_MRR']:.6f},{r['untrained_MRR']:.6f},"
                     f"{r['delta_MRR']:.6f},{r['trained_bits']:.4f},{r['untrained_bits']:.4f},{r['bits_added_over_untrained']:.4f},"
                     f"{r['trained_mixture_MRR']:.6f},{r['untrained_pool']},{r['trained_pool']},{r['untrained_seeds']},"
                     f"{r['status']},{r['trained_MRR_3seed']:.6f},{r['trained_MRR_source']}\n")
    run.set_summary({"verdicts": {"floor_gap": v_a, "floor_ordering": v_b},
                     "measured": {"floor_gap_models_delta_at_least_0.15": f"{n_ok} of {len(rows)}",
                                  "floor_gap_delta_by_model": {r["model"]: round(r["delta_MRR"], 4) for r in rows},
                                  "floor_ordering_spearman_trained_vs_delta": round(rho, 4) if rho == rho else None,
                                  "untrained_MRR_by_model": {r["model"]: round(r["untrained_MRR"], 4) for r in rows},
                                  "bits_added_over_untrained_by_model": {r["model"]: round(r["bits_added_over_untrained"], 3) for r in rows},
                                  "untrained_MRR_median": round(statistics.median(r["untrained_MRR"] for r in rows), 4) if rows else None,
                                  "floor_gap_at_floor": {m: AT_FLOOR[m] for m in AT_FLOOR if any(r["model"] == m for r in rows)},
                                  "floor_gap_reconciled_17sep": "trained side from the five-seed atlas row; ModernBERT-base mean-pooled with its mean-pooled twin",
                                  "floor_gap_superseded_rows": {r["model"]: {"pooling": r["pooling"], "untrained_MRR": round(r["untrained_MRR"], 4),
                                                                       "trained_MRR": round(r["trained_MRR"], 4), "delta_MRR": round(r["delta_MRR"], 4)} for r in superseded}}})
    run.flush()
    print(f"=== SUMMARY {NAME} collect ===", flush=True)
    print(f"floor_gap  {v_a}   delta at least 0.15 on {n_ok} of {len(rows)}", flush=True)
    print(f"floor_ordering  {v_b}   Spearman(trained MRR, delta) {rho:.3f}", flush=True)
    for r in sorted(rows, key=lambda r: -r["trained_MRR"]):
        print(f"  {r['model']:44s} trained {r['trained_MRR']:.4f} untrained {r['untrained_MRR']:.4f} delta "
              f"{r['delta_MRR']:+.4f} bits added {r['bits_added_over_untrained']:+.2f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
