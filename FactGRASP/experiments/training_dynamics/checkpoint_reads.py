# figure 4a and the falcon-mamba numbers (0.433 vs 0.427)
# for one model or checkpoint (repo@revision) at one layer: train its own linear map (native) and read it
# with the fixed llama-3.1-8b map through the random composition alignment (transferred)
# olmo-2-1b, once per pretraining checkpoint at layer 16:
#   python -m experiments.training_dynamics.checkpoint_reads --model allenai/OLMo-2-0425-1B@stage1-step10000-tokens21B
#       --layer 16 --floor-model allenai/OLMo-2-0425-1B --floor-tag olmo2_coarse --tag olmo2_checkpoints --tokens-B 21
# falcon-mamba: run the ladder and controls/untrained_floor with --tag falcon_mamba first, then
#   python -m experiments.training_dynamics.checkpoint_reads --model tiiuae/falcon-mamba-7b --floor-tag falcon_mamba --tag falcon_mamba

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

from factgrasp import crossmodel as X, data as D, encoders as E, gpu_guard, layers as L, metrics as M, paths, registry
from experiments.transfer.transfer_all_pairs import ModelSpace
from factgrasp.random_composition import field_rows, random_pair_rows

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "checkpoint_reads"
RUNS = os.path.join(paths.ROOT, "results", "runs")
FIELDS = ("subject", "pred_value", "object")
HUB, HUB2 = "meta-llama/Llama-3.1-8B", "Qwen/Qwen3.5-9B"


def build_space(model_id: str, layer: int, pooling: str, splits: dict, device, seeds: list, batch: int, cache_only: bool) -> SimpleNamespace:
    tok_model = [None, None]
    emb = {}
    for s in ("train", "valid", "test"):
        emb[s] = {}
        for f in FIELDS:
            p = E.cache_path(model_id, "wikidata", s, f, pooling, layer)
            if not os.path.exists(p):
                if cache_only:
                    raise SystemExit(f"[{NAME}] cache missing for {model_id} {s}/{f} at layer {layer} and --cache-only is set")
                if tok_model[0] is None:
                    print(f"[{NAME}] {model_id}: embedding at layer {layer} (loading weights)", flush=True)
                    tok_model[0], tok_model[1] = E.load_decoder(model_id, device)
            emb[s][f] = E.load_or_embed(splits[s][f].astype(str).tolist(), tok_model[0], tok_model[1], [layer], pooling,
                                        model_id=model_id, dataset="wikidata", split=s, field=f, device=device,
                                        batch=batch, progress=False)[layer].float()
    if tok_model[1] is not None:
        del tok_model[1]; torch.cuda.empty_cache()
    cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
    proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
    P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
    T = {s: {f: P(emb[s][f]) for f in FIELDS} for s in emb}
    x = {s: T[s]["subject"] + T[s]["pred_value"] for s in T}
    import pandas as pd
    valid_map = D.valid_sets(pd.concat([splits[s] for s in ("train", "valid", "test")], ignore_index=True))
    names, pool, gold = C.build_unique_object_pool(splits["test"], T["test"]["object"])
    vlists = D.valid_index_lists(splits["test"], names, valid_map)
    per_seed, probe0 = [], None
    for seed in seeds:
        m = L.train_single_map(x["train"], T["train"]["pred_value"], T["train"]["object"], x["valid"], T["valid"]["pred_value"],
                               T["valid"]["object"], device, loss="infonce", epochs=60, temp=0.05, seed=seed)
        o_hat, _ = L.predict(m, x["test"], T["test"]["pred_value"], device)
        res = M.evaluate(o_hat, pool, gold, valid=vlists, tie="random", seed=seed)
        per_seed.append({"seed": seed, "MRR": float(res["MRR"]), "Hits@1": float(res["Hits@1"]), "Hits@10": float(res["Hits@10"])})
        if seed == seeds[0]:
            probe0 = m
        else:
            del m
    sp = SimpleNamespace(model_id=model_id, layer=layer, pooling=pooling, width=int(cat.shape[1]),
                         x_train_cpu=x["train"], r_train=T["train"]["pred_value"], o_train=T["train"]["object"],
                         x_test=x["test"], r_test=T["test"]["pred_value"], pool_cpu=pool, gold=gold, _vl=vlists,
                         probe=probe0, per_seed=per_seed, mrr_native=statistics.mean(r["MRR"] for r in per_seed),
                         mrr_native_seed0=per_seed[0]["MRR"])
    sp.vlists = lambda splits, _v=vlists: _v
    return sp


def twin_space(floor_model: str, layer: int, pooling: str) -> SimpleNamespace:
    rid = floor_model + "-randinit"
    emb = {s: {f: torch.load(E.cache_path(rid, "wikidata", s, f, pooling, layer), map_location="cpu").float()
               for f in FIELDS} for s in ("train", "test")}
    cat = torch.cat([emb["train"][f] for f in FIELDS], 0)
    proj = E.PCAProjector.fit(cat, k=768, seed=0) if cat.shape[1] > 768 else None
    P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
    return SimpleNamespace(x_train_cpu=P(emb["train"]["subject"]) + P(emb["train"]["pred_value"]),
                           x_test=P(emb["test"]["subject"]) + P(emb["test"]["pred_value"]))


def floor_row(floor_model: str, floor_tag: str) -> dict:
    p = os.path.join(RUNS, f"untrained_floor_{floor_tag}.json")
    if not os.path.exists(p):
        return {}
    for v in (json.load(open(p)).get("rows") or {}).values():
        if isinstance(v, dict) and v.get("model") == floor_model and "untrained_MRR" in v:
            return v
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="repo or repo@revision")
    ap.add_argument("--layer", type=int, default=None, help="hidden-state layer; default the model's ladder JSON best_layer")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--floor-model", default=None, help="the model id untrained floor was run on (default: the repo without revision)")
    ap.add_argument("--floor-tag", required=True, help="the untrained floor registry tag holding the untrained twin row")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tokens-B", type=float, default=None, help="olmo checkpoints: the checkpoint's pretraining tokens, recorded in the row")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, f"{args.tag}__{E.model_slug(args.model)}", config=vars(args), thresholds={}, out_dir=RUNS)
    repo, revision = E.split_revision(args.model)
    floor_model = args.floor_model or repo
    layer = args.layer
    if layer is None:
        lad = os.path.join(paths.RUNS, f"ladder_{E.model_slug(args.model)}.json")
        layer = int(json.load(open(lad))["summary"]["best_layer"])
    key = registry.config_key({"model": args.model, "layer": layer, "pooling": args.pooling, "ridge": args.ridge})
    if run.done(key):
        print(f"[{NAME}] {args.model} at layer {layer}: already done", flush=True); return 0
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    t0 = time.time()
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    seeds = [int(s) for s in args.seeds.split(",")]
    sp = build_space(args.model, layer, args.pooling, splits, device, seeds, args.batch, args.cache_only)
    print(f"[{NAME}] {args.model} layer {layer}: native single map {sp.mrr_native:.4f} over {len(seeds)} seeds "
          f"({', '.join(f'{r[chr(77)+chr(82)+chr(82)]:.4f}' for r in sp.per_seed)}); {time.time() - t0:.0f}s", flush=True)
    fr = floor_row(floor_model, args.floor_tag)
    native_floor = fr.get("untrained_MRR")
    if fr and int(fr.get("layer", layer)) != layer:
        print(f"[{NAME}] note: the untrained floor floor row is at layer {fr.get('layer')}, this read is at {layer}", flush=True)
    twin = None
    try:
        twin = twin_space(floor_model, int(fr.get("layer", layer)), fr.get("pooling", args.pooling)) if fr else None
    except FileNotFoundError as exc:
        print(f"[{NAME}] twin caches missing ({exc}); hub floor left empty", flush=True)
    cpu = torch.device("cpu")
    hubs = {h: ModelSpace(h, splits, cpu, SimpleNamespace(temp=0.05, epochs=60)) for h in (HUB, HUB2)}
    n_pairs = int(hubs[HUB].x_train_cpu.shape[0])
    fr_m = field_rows(sp, splits)
    pairs_m = random_pair_rows(fr_m, n_pairs, seed=0)
    out = {"model": args.model, "repo": repo, "revision": revision, "tokens_B": args.tokens_B, "layer": layer, "pooling": args.pooling,
           "width": sp.width, "native_MRR": sp.mrr_native, "native_MRR_seed0": sp.mrr_native_seed0, "native_per_seed": sp.per_seed,
           "native_floor": native_floor, "floor_model": floor_model, "floor_tag": args.floor_tag, "hubs": {}}
    sp.x_test = sp.x_test.to(cpu)
    for h, hs in hubs.items():
        S_fwd = X.fit_stitch(sp.x_train_cpu, hs.x_train_cpu, ridge=args.ridge)
        S_rev = X.fit_stitch(hs.x_train_cpu, sp.x_train_cpu, ridge=args.ridge)
        S_rp = X.fit_stitch(pairs_m, random_pair_rows(field_rows(hs, splits), n_pairs, seed=0), ridge=args.ridge)
        mrr_hub, v = hs.stitched(sp, S_fwd, splits)
        mrr_zl, _ = hs.stitched(sp, S_rp, splits)
        matched = M.evaluate(v.cpu() @ S_rev, sp.pool_cpu, sp.gold, valid=sp.vlists(splits), tie="random", seed=0)
        hub_floor = None
        if twin is not None:
            S_tw = X.fit_stitch(twin.x_train_cpu, hs.x_train_cpu, ridge=args.ridge)
            hub_floor, _ = hs.stitched(twin, S_tw, splits)
        out["hubs"][h] = {"hub_native_MRR": hs.mrr_native, "hub_MRR": mrr_hub, "ratio": mrr_hub / sp.mrr_native,
                          "hub_MRR_zero_label": mrr_zl, "ratio_zero_label": mrr_zl / sp.mrr_native,
                          "hub_matched_own_pool": float(matched["MRR"]), "ratio_matched": float(matched["MRR"]) / sp.mrr_native,
                          "hub_floor": hub_floor,
                          "hub_gain_above_floor": (mrr_hub - hub_floor) if hub_floor is not None else None,
                          "native_gain_above_floor": (sp.mrr_native - native_floor) if native_floor is not None else None}
        o = out["hubs"][h]
        print(f"[{NAME}] {args.model} via {h.split('/')[-1]}: hub {mrr_hub:.4f} ({o['ratio']:.3f}) zero-label {mrr_zl:.4f} "
              f"({o['ratio_zero_label']:.3f}) matched {matched['MRR']:.4f} ({o['ratio_matched']:.3f}) hub floor "
              f"{(hub_floor if hub_floor is not None else float('nan')):.4f}; native {sp.mrr_native:.4f} floor "
              f"{(native_floor if native_floor is not None else float('nan')):.4f}", flush=True)
    out["seconds"] = round(time.time() - t0)
    run.put(key, out); run.flush()
    print(f"written: {run.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
