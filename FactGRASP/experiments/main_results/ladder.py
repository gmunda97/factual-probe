# table 2 table 3 table 5 table 6 come from here
# one frozen model: embed subject relation and object strings at a grid of layers, pca to 768, sweep a
# linear map per layer on valid mrr and pick the best layer, then train the 4 configs (linear map and 8 map MoT,
# cosine and cosine+infonce) at that layer and at the last layer with seeds 0 1 2 and test them
# the 19 paper models were run through all_models.py which calls this with the fixed protocol
# run: python -m experiments.main_results.ladder --model Qwen/Qwen3.5-4B --seeds 0,1,2
# the LOES layer selector is only logged here as a diagnostic, the paper uses the valid mrr pick

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from typing import Optional

import torch

from factgrasp import paths
from factgrasp import gpu_guard, registry
from factgrasp import data as D
from factgrasp import metrics as M
from factgrasp import encoders as E
from factgrasp import layers as L
from factgrasp import probes
from factgrasp import theory_reference as T

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

NAME = "ladder"
FIELDS = L.FIELDS

THRESHOLDS = {
    "level_green_MRR": 0.20,
    "level_amber_MRR": 0.10,
    "level_best_vs_last_relative": 0.10,
    "depth_lambda_star_range": [0.45, 0.90],
    "depth_depth_final_layer_within": 0.05,
    "depth_selector_regret_regret_green": 0.02,
    "depth_selector_regret_regret_red": 0.05,
}

LADDER_ROWS = [
    ("vanilla_fullwidth", "none", "none", "full"),
    ("vanilla_pca768", "none", "none", "pca"),
    ("single_cosine", "single", "cosine", "pca"),
    ("single_infonce", "single", "infonce", "pca"),
    ("mixture8_cosine", "mixture", "cosine", "pca"),
    ("mixture8_infonce", "mixture", "infonce", "pca"),
    ("single_infonce_fullwidth", "single", "infonce", "full"),
]


def json_safe(d: dict) -> dict:
    return {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in d.items() if not torch.is_tensor(v)}


def acquire_device(allow_cpu: bool):
    try:
        return gpu_guard.acquire()
    except SystemExit:
        if allow_cpu:
            print("[ladder] no GPU available, falling back to CPU (--allow-cpu)")
            return torch.device("cpu")
        raise


def maybe_yield(rr_store: dict, rr_path: str) -> None:
    try:
        yielding = bool(gpu_guard.check_yield())
    except Exception:
        yielding = False
    if yielding:
        save_rr(rr_store, rr_path)
        print("[ladder] yielding the GPU; state is checkpointed, rerun to resume")
        sys.exit(4)


def save_rr(rr_store: dict, path: str) -> None:
    if not rr_store:
        return
    tmp = path + ".tmp"
    torch.save(rr_store, tmp)
    os.replace(tmp, path)


def load_rr(path: str) -> dict:
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return {}
    return {}


def finalize(run, config: dict, summary: dict, verdict: str, notes: list) -> None:
    path = run.out_path
    blob = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            blob = {}
    rows = blob.get("rows", {k: v for k, v in blob.items()
                             if k not in ("name", "tag", "config", "summary",
                                          "thresholds", "verdict", "notes")})
    out = {"name": NAME, "tag": getattr(run, "tag", ""), "config": config,
           "rows": rows, "summary": summary, "thresholds": THRESHOLDS,
           "verdict": verdict, "notes": notes}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(tmp, path)


def embed_fields(df, tok, model, layers, pooling, model_id, dataset, split,
                 batch, max_len, device, frame="final") -> dict:
    out = {}
    for f in FIELDS:
        out[f] = E.load_or_embed(df[f].tolist(), tok, model, layers, pooling,
                                 model_id=model_id, dataset=dataset, split=split,
                                 field=f, batch=batch, max_len=max_len,
                                 device=device, frame=frame)
    return out


def stacked_queries(emb: dict, layers) -> torch.Tensor:
    cols = [(emb["subject"][l].float() + emb["pred_value"][l].float()).unsqueeze(1)
            for l in layers]
    return torch.cat(cols, dim=1)


def sweep_layer_cached(run, cache, layer, pooling, args, device, rr_store, rr_path):
    key = registry.config_key({"stage": 1, "what": "sweep", "pooling": pooling,
                               "layer": int(layer), "pca": args.pca,
                               "loss": args.infonce_loss, "epochs": args.epochs,
                               "seeds": args.sweep_seeds, "smoke": args.smoke})
    if run.done(key):
        return float(run.get(key)["MRR"])
    maybe_yield(rr_store, rr_path)
    t0 = time.time()
    mrr, det = L.sweep(cache, [layer], D_target=args.pca,
                       seeds=list(range(args.sweep_seeds)), loss=args.infonce_loss,
                       device=device, epochs=args.epochs,
                       batch_size=args.train_batch, temp=args.temp,
                       return_details=True)
    row = json_safe(det[layer])
    row["MRR"] = float(mrr[layer])
    row["seconds"] = round(time.time() - t0, 1)
    run.put(key, row)
    print(f"[stage1] pooling={pooling} layer={layer} val filtered MRR={row['MRR']:.4f}")
    return row["MRR"]


def stage1(run, args, tok, model, device, info, dfs, valid_map, rr_store, rr_path):
    n_layers = info["n_layers"]
    grid = args.grid
    df_calib, df_valid = dfs["calib"], dfs["valid"]

    sweeps, loes_all, caches = {}, {}, {}
    for pooling in args.poolings:
        emb_tr = embed_fields(df_calib, tok, model, grid, pooling, args.model,
                              args.dataset, dfs["calib_split_tag"], args.batch,
                              args.max_len, device, args.frame)
        emb_va = embed_fields(df_valid, tok, model, grid, pooling, args.model,
                              args.dataset, dfs["valid_split_tag"], args.batch,
                              args.max_len, device, args.frame)
        cache = L.SweepCache(train=emb_tr, valid=emb_va, df_train=df_calib,
                             df_valid=df_valid, valid_map=valid_map)
        caches[pooling] = cache

        loes_key = registry.config_key({"stage": 1, "what": "loes",
                                        "pooling": pooling, "layers": grid,
                                        "k": args.k_loes, "smoke": args.smoke})
        if run.done(loes_key):
            loes_all[pooling] = run.get(loes_key)
        else:
            q = stacked_queries(emb_va, grid).to(args.loes_device)
            tgt_last = emb_va["object"][grid[-1]].float().to(args.loes_device)
            tgt_same = [emb_va["object"][l].float().to(args.loes_device) for l in grid]
            res = L.loes_select(q, targets=tgt_last, k=args.k_loes, layers=grid,
                                targets_per_layer=tgt_same)
            del q, tgt_last, tgt_same
            loes_all[pooling] = res
            run.put(loes_key, res)
            print(f"[stage1] LOES[{pooling}] label-free={res['label_free']} "
                  f"supervised(last-layer target)={res['supervised']} "
                  f"supervised(same-layer target)={res['supervised_same_layer']}")

        sweeps[pooling] = {l: sweep_layer_cached(run, cache, l, pooling, args,
                                                 device, rr_store, rr_path)
                           for l in grid}

    per_pooling_best = {p: L.best_layer(s) for p, s in sweeps.items()}
    scores = {p: sweeps[p][per_pooling_best[p]] for p in sweeps}
    pooling = args.force_pooling or max(scores, key=scores.get)
    print(f"[stage1] pooling rule: {scores} -> chosen pooling = {pooling}")

    extra = [l for l in E.refine(per_pooling_best[pooling], n_layers)
             if l not in sweeps[pooling]]
    if extra:
        cache = caches[pooling]
        for split_key, df in (("calib", dfs["calib"]), ("valid", dfs["valid"])):
            tag = dfs[f"{split_key}_split_tag"]
            emb = embed_fields(df, tok, model, extra, pooling, args.model,
                               args.dataset, tag, args.batch, args.max_len, device,
                               args.frame)
            target = cache.train if split_key == "calib" else cache.valid
            for f in FIELDS:
                target[f].update(emb[f])
        for l in extra:
            sweeps[pooling][l] = sweep_layer_cached(run, cache, l, pooling, args,
                                                    device, rr_store, rr_path)

    mrr_by_layer = sweeps[pooling]
    best = L.best_layer(mrr_by_layer)
    loes = loes_all[pooling]
    depth = L.depth_report(mrr_by_layer, n_layers,
                           loes_label_free_top1=loes["label_free"][0],
                           loes_supervised_top1=(loes["supervised"][0]
                                                 if loes["supervised"] else None))

    chosen = []
    for cand in (best, n_layers, loes["label_free"][0]):
        if cand not in chosen:
            chosen.append(int(cand))
    ranked = sorted(mrr_by_layer, key=lambda l: -mrr_by_layer[l])
    for cand in ranked:
        if len(chosen) >= 3:
            break
        if cand not in chosen:
            chosen.append(int(cand))
    chosen = sorted(chosen)[:3] if len(chosen) >= 3 else sorted(chosen)

    decision = {"pooling": pooling, "pooling_scores": scores,
                "per_pooling_best_layer": per_pooling_best,
                "sweep": {p: {str(k): v for k, v in s.items()} for p, s in sweeps.items()},
                "loes": loes_all, "depth": depth, "chosen_layers": chosen,
                "best_layer": int(best), "n_layers": int(n_layers)}
    run.put(registry.config_key({"stage": 1, "what": "decision",
                                 "smoke": args.smoke}), decision)
    return decision


def build_probe(kind: str, dim: int, gate_dim: int, experts: int, device,
                out_dim: Optional[int] = None):
    out_dim = dim if out_dim is None else out_dim
    if kind == "single":
        m = V.FlexMixtureOfW(dim, out_dim, 1, gate_dim=gate_dim, gate_type="softmax")
    elif kind == "mixture":
        m = V.FlexMixtureOfW(dim, out_dim, experts, gate_dim=gate_dim, gate_type="softmax")
    else:
        raise ValueError(kind)
    return m.to(device)


def run_row(run, args, row, tensors, device, rr_store, rr_path, layer, pooling=None):
    name, kind, loss, width = row
    x_tr, hr_tr, o_tr = tensors["train"]
    x_va, hr_va, o_va = tensors["valid"]
    x_te, hr_te, o_te = tensors["test"]
    pool, gold, valid_lists = tensors["pool"], tensors["gold"], tensors["valid_lists"]

    seeds = [0] if kind == "none" else args.seeds
    per_seed, rrs, gates, n_params = [], [], None, 0
    for s in seeds:
        key = registry.config_key({"stage": 2, "row": name, "layer": int(layer),
                                   "seed": int(s), "pca": args.pca,
                                   "experts": args.experts, "epochs": args.epochs,
                                   "width": width, "smoke": args.smoke,
                                   "infonce_loss": args.infonce_loss})
        rr_key = f"{name}|L{layer}|seed{s}"
        if run.done(key) and rr_key in rr_store:
            row_res = run.get(key)
            per_seed.append(row_res)
            rrs.append(rr_store[rr_key])
            n_params = int(row_res.get("n_params", 0))
            continue
        maybe_yield(rr_store, rr_path)
        t0 = time.time()
        if kind == "none":
            o_hat = x_te
            n_params = 0
            g = None
        else:
            probe = build_probe(kind, x_tr.shape[1], hr_tr.shape[1], args.experts, device,
                                out_dim=o_tr.shape[1])
            n_params = probes.count_params(probe)
            loss_name = args.infonce_loss if loss == "infonce" else loss
            probe = V.train_minibatch(probe, x_tr, hr_tr, o_tr, x_va, hr_va, o_va,
                                      device, V.make_loss(loss_name, temp=args.temp),
                                      num_epochs=args.epochs, lr=args.lr,
                                      batch_size=args.train_batch,
                                      patience=args.patience,
                                      is_mixture=(kind == "mixture"))
            o_hat, g = L.predict(probe, x_te, hr_te, device)
            if kind == "mixture" and g is not None:
                gates = g
            if name == "mixture8_infonce":
                proj = tensors.get("proj")
                ckpt_dir = os.path.join(paths.CACHE_DEC, args.tag)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"probe_E{args.experts}_infonce_L{int(layer)}_seed{int(s)}.pt")
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in probe.state_dict().items()},
                            "E": int(args.experts), "D": int(x_tr.shape[1]), "layer": int(layer),
                            "seed": int(s), "pooling": pooling, "loss": "infonce",
                            "pca": ({"components": proj.components.cpu(), "mean": proj.mean.cpu()}
                                    if proj is not None else None),
                            "gate_type": "softmax", "use_shared": False,
                            "model": args.model, "dataset": args.dataset}, ckpt_path)
                print(f"[stage2] saved probe checkpoint {ckpt_path}")
        res = M.evaluate(o_hat, pool, gold, valid=valid_lists, tie=args.tie,
                         tau=args.temp)
        rr = res.get("rr")
        if rr is None:
            rr = 1.0 / T.ranks_from_scores(
                torch.nn.functional.normalize(o_hat.float(), dim=1)
                @ torch.nn.functional.normalize(pool.float(), dim=1).t(),
                gold, valid_lists, "optimistic").float()
        rr_store[rr_key] = rr.float().cpu()
        rrs.append(rr_store[rr_key])
        row_res = json_safe(res)
        row_res.update({"n_params": n_params, "layer": int(layer), "seed": int(s),
                        "row": name, "loss": loss, "width": width,
                        "dim": int(x_tr.shape[1]), "seconds": round(time.time() - t0, 1)})
        if kind == "mixture" and g is not None:
            row_res.update(T.effective_experts(g))
        run.put(key, row_res)
        save_rr(rr_store, rr_path)
        per_seed.append(row_res)
        print(f"[stage2] L{layer} {name} seed={s} filtered MRR={row_res['MRR']:.4f} "
              f"({n_params:,} params)")

    numeric = [{k: float(v) for k, v in r.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
               for r in per_seed]
    agg = M.aggregate_seeds(numeric)
    agg["n_params"] = n_params
    agg["layer"] = int(layer)
    agg["row"] = name
    if gates is not None:
        agg["N_eff"] = T.effective_experts(gates)
    elif "N_eff_global" in per_seed[0]:
        agg["N_eff"] = {"N_eff_global": per_seed[0]["N_eff_global"],
                        "N_eff_per_input": per_seed[0]["N_eff_per_input"]}
    return agg, torch.stack(rrs, 0).mean(0)


def make_tensors(embs: dict, dfs: dict, layer: int, width: str, args, valid_map):
    proj = None
    qform = getattr(args, "query_form", "add")
    k_target = args.pca // 2 if qform == "concat" else args.pca
    if width == "pca":
        rows = torch.cat([embs["train"][f][layer].float() for f in FIELDS], dim=0)
        if rows.shape[1] > k_target:
            proj = E.PCAProjector.fit(rows, k=k_target, seed=0)
    out = {}
    for split in ("train", "valid", "test"):
        hs, hr, ho = L.project_split(embs[split], layer, proj)
        out[split] = (L.compose_query(hs, hr, qform), hr, ho)
    _, _, ho_te = out["test"]
    unique_names, pool, gold = C.build_unique_object_pool(dfs["test"], ho_te)
    out["pool"] = pool
    out["gold"] = gold
    out["valid_lists"] = (D.valid_index_lists(dfs["test"], unique_names, valid_map)
                          if valid_map is not None else None)
    out["unique_names"] = unique_names
    out["proj"] = proj
    return out


def rows_for(args):
    if getattr(args, "query_form", "add") == "add":
        return LADDER_ROWS
    return [r for r in LADDER_ROWS if r[1] != "none"]


def stage2(run, args, tok, model, device, info, dfs, valid_map, decision,
           rr_store, rr_path):
    pooling = decision["pooling"]
    chosen = [int(l) for l in decision["chosen_layers"]]
    best = int(decision["best_layer"])
    n_layers = int(info["n_layers"])

    embs = {}
    for split in ("train", "valid", "test"):
        embs[split] = embed_fields(dfs[split], tok, model, chosen, pooling,
                                   args.model, args.dataset,
                                   dfs[f"{split}_split_tag"], args.batch,
                                   args.max_len, device, args.frame)

    rows, rr_mean = {}, {}
    tensors_pca = make_tensors(embs, dfs, best, "pca", args, valid_map)
    tensors_full = make_tensors(embs, dfs, best, "full", args, valid_map)
    for row in rows_for(args):
        tensors = tensors_pca if row[3] == "pca" else tensors_full
        agg, rr = run_row(run, args, row, tensors, device, rr_store, rr_path, best,
                          pooling=pooling)
        rows[row[0]] = agg
        rr_mean[row[0]] = rr

    if n_layers != best:
        tensors_last = make_tensors(embs, dfs, n_layers, "pca", args, valid_map)
        agg, rr = run_row(run, args, ("single_infonce", "single", "infonce", "pca"),
                          tensors_last, device, rr_store, rr_path, n_layers,
                          pooling=pooling)
        rows["single_infonce_last_layer"] = agg
        rr_mean["single_infonce_last_layer"] = rr
    else:
        rows["single_infonce_last_layer"] = rows["single_infonce"]
        rr_mean["single_infonce_last_layer"] = rr_mean["single_infonce"]

    save_rr(rr_store, rr_path)
    return rows, rr_mean, embs


def _mean_of(agg: dict, key: str = "MRR"):
    v = agg.get(key)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def build_summary(rows, rr_mean, decision, info, args):
    gap_mixture = M.paired_ci(rr_mean["mixture8_infonce"], rr_mean["single_infonce"])
    gap_objective = M.paired_ci(rr_mean["single_infonce"], rr_mean["single_cosine"])
    mrr_head = _mean_of(rows["mixture8_infonce"])
    mrr_best = _mean_of(rows["single_infonce"])
    mrr_last = _mean_of(rows["single_infonce_last_layer"])
    rel_drop = ((mrr_best - mrr_last) / mrr_best) if (mrr_best and mrr_best > 0) else None

    depth = decision["depth"]
    lam = depth["lambda_star"]
    lo, hi = THRESHOLDS["depth_lambda_star_range"]
    regret_lf = depth.get("selector_regret_label_free")

    ordering_ok = (gap_mixture["diff"] > 0 and gap_objective["diff"] > 0)
    both_sig = gap_mixture["significant"] and gap_objective["significant"]
    depth_ok = rel_drop is not None and rel_drop >= THRESHOLDS["level_best_vs_last_relative"]

    if (mrr_head is not None and mrr_head >= THRESHOLDS["level_green_MRR"]
            and both_sig and ordering_ok and depth_ok):
        verdict = "green"
    elif (mrr_head is not None and mrr_head >= THRESHOLDS["level_amber_MRR"]
          and ordering_ok
          and (gap_mixture["significant"] or gap_objective["significant"])):
        verdict = "amber"
    else:
        verdict = "red"

    summary = {
        "model": args.model, "dataset": args.dataset, "pooling": decision["pooling"],
        "n_layers": info["n_layers"], "hidden": info["hidden"],
        "best_layer": decision["best_layer"], "chosen_layers": decision["chosen_layers"],
        "headline_MRR_mixture_infonce": mrr_head,
        "MRR_single_infonce": mrr_best,
        "MRR_single_infonce_last_layer": mrr_last,
        "best_vs_last_relative": rel_drop,
        "gap_mixture_vs_single": gap_mixture,
        "gap_infonce_vs_cosine": gap_objective,
        "N_eff": rows["mixture8_infonce"].get("N_eff"),
        "params": {k: rows[k].get("n_params") for k in rows},
        "lambda_star": lam,
        "depth_lambda_in_range": (lo <= lam <= hi) if lam is not None else None,
        "depth_not_final_layer": decision["best_layer"] != info["n_layers"],
        "depth_final_layer_10pct_below": depth_ok,
        "selector_regret_selector_regret_label_free": regret_lf,
        "selector_regret_green": (regret_lf is not None
                      and regret_lf <= THRESHOLDS["depth_selector_regret_regret_green"]),
        "verdict": verdict,
    }
    return summary, verdict


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="layer sweep and the four configs for one frozen model")
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dataset", default="wikidata")
    ap.add_argument("--query-form", default="add", choices=["add", "concat"],
                    help="how the subject and relation vectors are combined into the query. "
                         "concat halves the per-field projection so both forms give the same "
                         "query width and the same parameter count in W (add vs concat).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "1", "2"])
    ap.add_argument("--calib", type=int, default=8000)
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sweep-seeds", type=int, default=1,
                    help="probe seeds per sweep point (selection only)")
    ap.add_argument("--batch", type=int, default=256, help="embedding batch")
    ap.add_argument("--train-batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--infonce-loss", default="infonce",
                    choices=["infonce", "infonce_only"],
                    help="'infonce' is the composite cosine + InfoNCE of "
                         "variants.make_loss used by the BERT ladder; "
                         "'infonce_only' drops the cosine term")
    ap.add_argument("--tie", default="random")
    ap.add_argument("--k-loes", type=int, default=3)
    ap.add_argument("--force-pooling", default=None, choices=[None, "last", "mean"])
    ap.add_argument("--prefer", default="causal", choices=["causal", "auto"],
                    help="which Auto class to try first in load_decoder")
    ap.add_argument("--loes-on-cpu", action="store_true",
                    help="run the LOES ridge solves on CPU to save VRAM")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32", "int8", "nf4"],
                    help="int8 and nf4 load quantised weights with a bf16 compute dtype, "
                         "which is how the Tier-2 leftovers fit on one card")
    ap.add_argument("--grid-every", type=int, default=0,
                    help="sweep every Nth layer instead of encoders.layer_grid; the atlas "
                         "protocol uses 4. 0 keeps the Phase-0 grid.")
    ap.add_argument("--grid-layers", default="",
                    help="explicit comma separated sweep layers, overriding --grid-every")
    ap.add_argument("--grid-depths", default="",
                    help="comma separated relative depths, e.g. 0.5,0.8,1.0, for the "
                         "models above 12B that the atlas sweeps at three layers only")
    ap.add_argument("--frame", default="final", choices=["final", "raw"],
                    help="normalisation frame for intermediate hidden states. 'final' "
                         "passes every layer through the text tower's final norm, which "
                         "transformers already applies to the last hidden state only; "
                         "'raw' is the unnormalised residual stream, the protocol used "
                         "before 2026-09-07.")
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    args.poolings = ["last", "mean"]
    if args.tag is None:
        args.tag = (E.model_slug(args.model)
                    + ("" if args.frame == "final" else f"_{args.frame}")
                    + ("_smoke" if args.smoke else ""))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    C.set_seed(42)
    device = acquire_device(args.allow_cpu)
    args.loes_device = torch.device("cpu") if args.loes_on_cpu else device

    info = E.encoder_info(args.model)
    n_layers = info["n_layers"]
    args.grid = E.layer_grid(n_layers)
    if getattr(args, "grid_layers", ""):
        args.grid = sorted({min(max(1, int(x)), n_layers)
                            for x in args.grid_layers.split(",") if x.strip()})
    elif getattr(args, "grid_depths", ""):
        args.grid = sorted({max(1, min(n_layers, int(round(float(d) * n_layers))))
                            for d in args.grid_depths.split(",") if d.strip()})
    elif getattr(args, "grid_every", 0):
        step = int(args.grid_every)
        args.grid = sorted(set(list(range(step, n_layers + 1, step)) + [n_layers]))
    if args.smoke:
        args.grid = sorted({max(1, n_layers // 2), n_layers})
        args.epochs = 2
        args.calib = 512
        args.sweep_seeds = 1
        args.seeds = args.seeds[:2]
    print(f"[ladder] {args.model}: {n_layers} layers, hidden {info['hidden']}, "
          f"{info['heads']}q/{info['kv_heads']}kv heads, grid {args.grid}")

    df_train = D.load_dataset(args.dataset, "train")
    df_valid = D.load_dataset(args.dataset, "valid")
    df_test = D.load_dataset(args.dataset, "test")
    if args.smoke:
        df_train = df_train.head(512).reset_index(drop=True)
        df_valid = df_valid.head(512).reset_index(drop=True)
        df_test = df_test.head(512).reset_index(drop=True)
    import pandas as pd
    valid_map = D.valid_sets(pd.concat([df_train, df_valid, df_test], ignore_index=True))

    n_cal = min(args.calib, len(df_train))
    df_calib = df_train.sample(n=n_cal, random_state=0).reset_index(drop=True)
    dfs = {
        "train": df_train, "valid": df_valid, "test": df_test, "calib": df_calib,
        "train_split_tag": "train" + ("512" if args.smoke else ""),
        "valid_split_tag": "valid" + ("512" if args.smoke else ""),
        "test_split_tag": "test" + ("512" if args.smoke else ""),
        "calib_split_tag": f"traincal{n_cal}",
    }

    run = registry.Run(NAME, args.tag)
    rr_path = os.path.join(paths.RUNS, f"{NAME}_{args.tag}_rr.pt")
    rr_store = load_rr(rr_path)

    tok, model = E.load_decoder(args.model, device, dtype=args.dtype,
                                prefer=args.prefer)
    tm_name = type(E.text_model(model)).__name__
    print(f"[ladder] loaded {type(model).__name__}; text tower = {tm_name} "
          f"with {len(E.decoder_layers(model))} blocks")

    notes = [
        f"text tower resolved to {tm_name}",
        "pooling rule: higher validation filtered MRR for the single map under "
        "InfoNCE at that pooling's own sweep-best layer",
        "PCA-768 is fitted per layer on the train rows of subject, relation and "
        "object stacked; each field is projected before the query is composed",
        "layer indices are hidden-state indices (0 = embedding output)",
    ]

    dec_key = registry.config_key({"stage": 1, "what": "decision", "smoke": args.smoke})
    if args.stage == "2" and run.done(dec_key):
        decision = run.get(dec_key)
    else:
        decision = stage1(run, args, tok, model, device, info, dfs, valid_map,
                          rr_store, rr_path)
    if args.stage == "1":
        finalize(run, vars_config(args, info), {"stage1": decision}, "n/a", notes)
        print(json.dumps({"stage1_decision": decision["chosen_layers"],
                          "pooling": decision["pooling"]}, indent=2))
        return

    rows, rr_mean, _ = stage2(run, args, tok, model, device, info, dfs, valid_map,
                              decision, rr_store, rr_path)
    summary, verdict = build_summary(rows, rr_mean, decision, info, args)
    summary["stage1"] = {"pooling_scores": decision["pooling_scores"],
                         "sweep": decision["sweep"], "loes": decision["loes"],
                         "depth": decision["depth"]}
    finalize(run, vars_config(args, info), summary, verdict, notes)

    table = []
    for name in [r[0] for r in rows_for(args)] + ["single_infonce_last_layer"]:
        agg = rows.get(name)
        if agg is None:
            continue
        table.append({
            "row": name, "layer": agg.get("layer"),
            "MRR": _mean_of(agg), "Hits@1": _mean_of(agg, "Hits@1"),
            "Hits@10": _mean_of(agg, "Hits@10"), "params": agg.get("n_params"),
        })
    table.append({"row": "gap mixture-single", "MRR": summary["gap_mixture_vs_single"]["diff"],
                  "layer": decision["best_layer"], "Hits@1": None, "Hits@10": None,
                  "params": None})
    table.append({"row": "gap infonce-cosine", "MRR": summary["gap_infonce_vs_cosine"]["diff"],
                  "layer": decision["best_layer"], "Hits@1": None, "Hits@10": None,
                  "params": None})
    run.summary(table)
    print(f"VERDICT {args.model}: {verdict}")


def vars_config(args, info) -> dict:
    cfg = {k: v for k, v in vars(args).items() if k not in ("grid", "loes_device")}
    cfg["grid"] = args.grid
    cfg["encoder_info"] = info
    return cfg


if __name__ == "__main__":
    main()
