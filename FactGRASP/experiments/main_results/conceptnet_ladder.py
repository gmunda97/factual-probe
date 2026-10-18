# same 4 config ladder on conceptnet at each model's chosen layer, run for the 19 models
# its per triple ranks give the conceptnet number in appendix e (relation oracle gain 0.011)
# run: python -m experiments.main_results.conceptnet_ladder --model Qwen/Qwen3.5-4B

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from factgrasp import paths
from factgrasp import gpu_guard, registry
from factgrasp import data as D
from factgrasp import metrics as M
from factgrasp import encoders as E
from factgrasp import layers as L
from factgrasp import probes

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

NAME = "conceptnet_ladder"
FIELDS = ("subject", "pred_value", "object")
RUNS = os.path.join(paths.ROOT, "results", "runs")
os.makedirs(RUNS, exist_ok=True)

THRESHOLDS = {"graph_ordering_monotone_graphs_green": 4, "graph_ordering_monotone_graphs_amber": 3,
              "graph_ordering_models_needed_green": 2}
ORDER_GRAPHS = ["wikidata", "trex", "conceptnet", "fb15k237"]
ALL_GRAPHS = ["wikidata", "popqa", "trex", "conceptnet", "fb15k237"]
ROWS = [("single_cosine", 1, "cosine"), ("single_infonce", 1, "infonce"),
        ("mixture8_cosine", 8, "cosine"), ("mixture8_infonce", 8, "infonce")]


def ladder_summary(model_id: str) -> dict:
    tag = E.model_slug(model_id)
    path = os.path.join(paths.RUNS, f"ladder_{tag}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[{NAME}] missing ladder JSON {path}; run ladder first")
    return json.load(open(path))["summary"]


def cap_splits(splits: dict, cap: int, seed: int = 0) -> dict:
    total = sum(len(d) for d in splits.values())
    out = {}
    for sp, df in splits.items():
        k = max(1, int(round(cap * len(df) / total)))
        if k >= len(df):
            out[sp] = df.reset_index(drop=True)
            continue
        frac = k / len(df)
        keep = []
        for _, idx in df.groupby("pred_value").indices.items():
            n_take = max(1, int(round(frac * len(idx))))
            g = torch.Generator().manual_seed(seed + len(idx))
            sel = torch.randperm(len(idx), generator=g)[:n_take].tolist()
            keep.extend(int(idx[i]) for i in sel)
        picked = df.iloc[sorted(keep)]
        if len(picked) > k:
            picked = picked.sample(n=k, random_state=seed)
        out[sp] = picked.reset_index(drop=True)
    return out


def cache_ready(model_id, dataset, split, field, pooling, layer, n_rows: int) -> bool:
    p = E.cache_path(model_id, dataset, split, field, pooling, layer)
    if not os.path.exists(p):
        return False
    try:
        return int(torch.load(p, map_location="cpu").shape[0]) == int(n_rows)
    except Exception:
        return False


def split_tag(split: str, n_rows: int, full_rows: int) -> str:
    return split if n_rows == full_rows else f"{split}n{n_rows}"


def embed_dataset(model_id, dataset, splits, pooling, layer, tok, model, args, device,
                  tags: dict):
    out = {}
    for sp, df in splits.items():
        out[sp] = {}
        for f in FIELDS:
            texts = df[f].astype(str).tolist()
            got = E.load_or_embed(texts, tok, model, [layer], pooling, model_id=model_id,
                                 dataset=dataset, split=tags[sp], field=f,
                                 batch=args.batch, max_len=args.max_len, device=device,
                                 progress=False)
            out[sp][f] = got[layer].float()
    return out


def train_row(kind_experts, loss, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device, args, seed):
    C.set_seed(seed)
    dim = x_tr.shape[1]
    m = V.FlexMixtureOfW(dim, dim, kind_experts, gate_dim=dim, gate_type="softmax").to(device)
    m = V.train_minibatch(m, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device,
                          V.make_loss(loss, temp=args.temp),
                          num_epochs=2 if args.smoke else args.epochs,
                          batch_size=args.train_batch, patience=args.patience,
                          is_mixture=kind_experts > 1)
    return m


def popularity_strata(df_test, rr: torch.Tensor) -> dict:
    path = os.path.join(paths.DATA_EXTRA, "popqa_popularity.csv")
    if not os.path.exists(path):
        return {"error": "popqa_popularity.csv not found"}
    import pandas as pd
    pop = pd.read_csv(path)
    col = next((c for c in pop.columns if "pop" in c.lower() and c != "subject"), None)
    key = "subject" if "subject" in pop.columns else pop.columns[0]
    if col is None:
        return {"error": f"no popularity column in {list(pop.columns)[:6]}"}
    lut = dict(zip(pop[key].astype(str), pop[col]))
    vals = torch.tensor([float(lut.get(str(s), float("nan")))
                         for s in df_test["subject"].astype(str)], dtype=torch.float)
    ok = torch.isfinite(vals)
    if int(ok.sum()) < 8:
        return {"error": f"only {int(ok.sum())} test subjects carry a popularity value"}
    q = torch.quantile(vals[ok], torch.tensor([0.25, 0.5, 0.75]))
    out = {"n_with_popularity": int(ok.sum()), "cuts": [float(v) for v in q]}
    for i in range(4):
        lo = -float("inf") if i == 0 else float(q[i - 1])
        hi = float("inf") if i == 3 else float(q[i])
        m = ok & (vals > lo) & (vals <= hi) if i else ok & (vals <= float(q[0]))
        out[f"quartile_{i + 1}"] = {"n": int(m.sum()),
                                    "MRR": float(rr[m].mean()) if int(m.sum()) else float("nan")}
    return out


def cap_pool(df_test, cap: int, seed: int = 0):
    import numpy as np
    objs = df_test["object"].tolist()
    order = np.random.RandomState(seed).permutation(len(objs))
    admitted, keep = set(), []
    for i in order:
        o = objs[i]
        if o in admitted:
            keep.append(int(i))
        elif len(admitted) < cap:
            admitted.add(o)
            keep.append(int(i))
    keep.sort()
    return keep, len(admitted)


def main() -> int:
    ap = argparse.ArgumentParser(description="the four config ladder on conceptnet at a model's chosen layer")
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="conceptnet")
    ap.add_argument("--pooling", default="")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train-batch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--tie", default="random")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cap-rows", type=int, default=30000,
                    help="total rows for the datasets named by --cap-datasets, before a "
                         "seeded stratified subsample. 0 disables the cap.")
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="subsample the test rows of every dataset until their distinct "
                         "objects number this many, so the candidate pool is the same size "
                         "on every graph. 0 disables it. graph_ordering compares four graphs whose pools "
                         "differ by a factor of 4.5, and MRR depends on pool size for reasons "
                         "unrelated to answer multiplicity; this holds it fixed.")
    ap.add_argument("--cap-datasets", default="fb15k237",
                    help="comma separated. a 30k subsample is used for "
                         "FB15k-237 only; every other graph is used whole, so that the "
                         "Wikidata row stays identical to the Phase-0 one.")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--prefer", default="causal")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    graphs = [g.strip() for g in args.datasets.split(",") if g.strip()]
    if not args.tag:
        args.tag = E.model_slug(args.model) + ("_smoke" if args.smoke else "")

    lad = ladder_summary(args.model)
    pooling = args.pooling or lad["pooling"]
    layer = args.layer if args.layer >= 0 else int(lad["best_layer"])

    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    run = registry.Run(NAME, args.tag, thresholds=THRESHOLDS, out_dir=RUNS)
    run.set_config(dict(vars(args), pooling=pooling, layer=layer, device=str(device)))
    rr_path = os.path.join(RUNS, f"{NAME}_{args.tag}_rr.pt")
    rr_store = torch.load(rr_path, map_location="cpu") if os.path.exists(rr_path) else {}

    print(f"[{NAME}] {args.model}: layer {layer}, pooling {pooling}, graphs {graphs}",
          flush=True)

    tok = model = None

    def ensure_model():
        nonlocal tok, model
        if model is None:
            print(f"[{NAME}] loading {args.model} to embed the missing graphs", flush=True)
            tok, model = E.load_decoder(args.model, device, dtype=args.dtype,
                                        prefer=args.prefer)
            print(f"[{NAME}] text tower {type(E.text_model(model)).__name__}", flush=True)
        return tok, model

    per_graph, measured, verdicts = {}, {}, {}
    for ds in graphs:
        key_ds = registry.config_key({"dataset": ds, "layer": layer, "pooling": pooling,
                                      "pool_cap": int(args.pool_cap),
                                      "seeds": seeds, "smoke": args.smoke})
        if run.done(key_ds):
            per_graph[ds] = run.get(key_ds)
            print(f"[{NAME}] {ds}: cached", flush=True)
            continue
        try:
            splits = {s: D.load_dataset(ds, s) for s in ("train", "valid", "test")}
            full_rows = {s: len(df) for s, df in splits.items()}
        except FileNotFoundError as exc:
            run.note(f"{ds} skipped: {exc}")
            print(f"[{NAME}] {ds} skipped: {exc}", flush=True)
            continue
        if args.smoke:
            splits = {s: df.head(512).reset_index(drop=True) for s, df in splits.items()}
        elif (args.cap_rows and ds in [c.strip() for c in args.cap_datasets.split(",")]
              and sum(len(d) for d in splits.values()) > args.cap_rows):
            splits = cap_splits(splits, args.cap_rows, seed=0)
            print(f"[{NAME}] {ds} capped to {sum(len(d) for d in splits.values())} rows "
                  f"(a 30k subsample is used for FB15k-237)", flush=True)

        tags = {s: split_tag(s, len(df), full_rows[s]) for s, df in splits.items()}
        need = any(not cache_ready(args.model, ds, tags[s], f, pooling, layer, len(df))
                   for s, df in splits.items() for f in FIELDS)
        if need:
            ensure_model()
        emb = embed_dataset(args.model, ds, splits, pooling, layer, tok, model, args,
                            device, tags)

        import pandas as pd
        valid_map = D.valid_sets(pd.concat(list(splits.values()), ignore_index=True))
        proj = None
        rows_cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
        if rows_cat.shape[1] > args.pca:
            proj = E.PCAProjector.fit(rows_cat, k=args.pca, seed=0)
        def P(t):
            return proj.transform(t) if proj is not None else t
        tr = {f: P(emb["train"][f]) for f in FIELDS}
        va = {f: P(emb["valid"][f]) for f in FIELDS}
        te = {f: P(emb["test"][f]) for f in FIELDS}
        x_tr, x_va, x_te = (tr["subject"] + tr["pred_value"],
                            va["subject"] + va["pred_value"],
                            te["subject"] + te["pred_value"])
        pool_capped = False
        if args.pool_cap and not args.smoke:
            keep, n_obj = cap_pool(splits["test"], args.pool_cap, seed=0)
            if n_obj < args.pool_cap:
                run.note(f"{ds}: pool cap {args.pool_cap} exceeds its {n_obj} distinct "
                         f"objects; evaluated uncapped")
                print(f"[{NAME}] {ds}: only {n_obj} distinct objects, below the cap of "
                      f"{args.pool_cap}; left uncapped", flush=True)
            else:
                idx = torch.tensor(keep, dtype=torch.long)
                splits["test"] = splits["test"].iloc[keep].reset_index(drop=True)
                te = {f: v[idx] for f, v in te.items()}
                x_te = x_te[idx]
                pool_capped = True
                print(f"[{NAME}] {ds}: test rows {len(idx)}, pool capped to {n_obj} "
                      f"distinct objects", flush=True)
        names, pool, gold = C.build_unique_object_pool(splits["test"], te["object"])
        vlists = D.valid_index_lists(splits["test"], names, valid_map)
        rho, m_counts = D.multiplicity_rate(splits["test"], names, valid_map)
        print(f"[{NAME}] {ds}: train {len(splits['train'])}, test {len(splits['test'])}, "
              f"pool {len(names)}, multiplicity rate {100 * rho:.2f} percent", flush=True)

        cells = {}
        for row_name, experts, loss in ROWS:
            per_seed, rrs = [], []
            for seed in seeds:
                m = train_row(experts, loss, x_tr, tr["pred_value"], tr["object"],
                              x_va, va["pred_value"], va["object"], device, args, seed)
                o_hat, _ = L.predict(m, x_te, te["pred_value"], device)
                res = M.evaluate(o_hat, pool, gold, valid=vlists, tie=args.tie,
                                 tau=args.temp, seed=seed)
                per_seed.append(M.jsonable(res))
                rrs.append(res["rr"])
                rr_store[f"{ds}|{row_name}|seed{seed}"] = res["rr"].cpu()
                del m
            agg = M.aggregate_seeds(per_seed)
            agg["n_params"] = probes.count_params(
                V.FlexMixtureOfW(x_tr.shape[1], x_tr.shape[1], experts,
                                 gate_dim=x_tr.shape[1]))
            cells[row_name] = agg
            cells[row_name + "_rr_mean"] = None
            print(f"[{NAME}] {ds:10s} {row_name:16s} MRR {agg['MRR']:.4f} "
                  f"+- {agg.get('MRR_ci95', float('nan')):.4f}", flush=True)
            if gpu_guard.check_yield():
                torch.save(rr_store, rr_path)
                run.note(f"yielded during {ds}/{row_name}")
                gpu_guard.yield_and_exit("rerun to continue")

        sc, si = cells["single_cosine"]["MRR"], cells["single_infonce"]["MRR"]
        mc, mi = cells["mixture8_cosine"]["MRR"], cells["mixture8_infonce"]["MRR"]
        rec = {
            "dataset": ds, "multiplicity_rate": rho, "pool_size": int(len(names)),
            "pool_capped": pool_capped, "pool_cap": int(args.pool_cap),
            "capped": bool(args.cap_rows and not args.smoke
                           and ds in [c.strip() for c in args.cap_datasets.split(",")]),
            "n_train": int(len(splits["train"])), "n_test": int(len(splits["test"])),
            "n_relations": int(splits["train"]["pred_value"].nunique()),
            "mean_valid_in_pool": float(m_counts.float().mean()),
            "cells": {k: v for k, v in cells.items() if not k.endswith("_rr_mean")},
            "gain_ratio_single": (si / sc) if sc > 0 else float("nan"),
            "gain_abs_single": si - sc,
            "gain_ratio_mixture": (mi / mc) if mc > 0 else float("nan"),
            "mixture_gain_under_cosine": mc - sc,
            "mixture_gain_under_infonce": mi - si,
            "paired_infonce_minus_cosine_single": M.paired_ci(
                M.mean_rr([rr_store[f"{ds}|single_infonce|seed{s}"] for s in seeds]),
                M.mean_rr([rr_store[f"{ds}|single_cosine|seed{s}"] for s in seeds]),
                n_boot=args.n_boot),
        }
        if ds == "popqa":
            rec["popularity"] = popularity_strata(
                splits["test"],
                M.mean_rr([rr_store[f"{ds}|mixture8_infonce|seed{s}"] for s in seeds]))
        run.put(key_ds, rec)
        torch.save(rr_store, rr_path)
        per_graph[ds] = rec

    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def ordering(names_list):
        pts = [(per_graph[g]["multiplicity_rate"], per_graph[g]["gain_ratio_single"], g)
               for g in names_list if g in per_graph
               and math.isfinite(per_graph[g].get("gain_ratio_single", float("nan")))]
        pts.sort()
        ratios = [p[1] for p in pts]
        in_order = sum(1 for a, b in zip(ratios, ratios[1:]) if b >= a - 1e-9)
        return {"graphs": [p[2] for p in pts],
                "multiplicity_rates": [p[0] for p in pts], "gain_ratios": ratios,
                "n_points": len(pts), "n_adjacent_in_order": in_order,
                "monotone": len(pts) > 1 and in_order == len(pts) - 1}

    four = ordering(ORDER_GRAPHS)
    five = ordering(ALL_GRAPHS)
    summary = {"model": args.model, "layer": layer, "pooling": pooling,
               "graphs_measured": sorted(per_graph),
               "graph_ordering_four_graph_ordering": four, "five_graph_ordering": five,
               "per_graph": {g: {k: v for k, v in r.items() if k != "cells"}
                             for g, r in per_graph.items()}}
    if four["n_points"] >= 2:
        verdicts["graph_ordering"] = ("green" if four["monotone"]
                          else "amber" if four["n_adjacent_in_order"] >= THRESHOLDS[
                              "graph_ordering_monotone_graphs_amber"] - 1 else "red")
        measured["graph_ordering"] = ("gain ratio by rising multiplicity over "
                          + ", ".join(f"{g} {100 * r:.1f} percent -> {x:.3f}"
                                      for g, r, x in zip(four["graphs"],
                                                         four["multiplicity_rates"],
                                                         four["gain_ratios"]))
                          + f"; monotone {four['monotone']}")
    summary["verdicts"] = verdicts
    summary["measured"] = measured
    run.set_summary(summary)
    run.set_verdict(verdicts.get("graph_ordering", "n/a"))
    run.note("the ordering is read on four graphs, the five graph ordering with PopQA is reported beside it")

    srows = []
    for g in ALL_GRAPHS:
        r = per_graph.get(g)
        if not r:
            continue
        srows.append({"dataset": g, "rho_pct": 100 * r["multiplicity_rate"],
                      "single_cos": r["cells"]["single_cosine"]["MRR"],
                      "single_nce": r["cells"]["single_infonce"]["MRR"],
                      "ratio": r["gain_ratio_single"],
                      "mix_gain_cos": r["mixture_gain_under_cosine"],
                      "mix_gain_nce": r["mixture_gain_under_infonce"]})
    if "graph_ordering" in measured:
        srows.append({"dataset": "graph_ordering", "rho_pct": verdicts["graph_ordering"],
                      "single_cos": "monotone", "single_nce": four["monotone"],
                      "ratio": f"{four['n_adjacent_in_order']}/{max(four['n_points'] - 1, 1)}",
                      "mix_gain_cos": "-", "mix_gain_nce": "-"})
    run.summary(srows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
