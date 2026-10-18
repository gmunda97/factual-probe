# intervention protocol behind table 12: add the map's predicted direction into the residual stream
# and measure the effect against a matched random direction. wrap_pca and read_ladder are imported elsewhere

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from factgrasp import paths

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

from factgrasp import causal, data, encoders, gpu_guard, probes, registry
from factgrasp import theory_reference as T

NAME = "steering"
FIELDS = ("subject", "pred_value", "object")

THRESHOLDS = {
    "gain_nats_green": 0.5,
    "monotone_min_steps": 3,
    "decode_rate_green": 0.25,
    "decode_rate_amber": 0.10,
    "decode_rate_random_max": 0.05,
}


def _dig(blob: dict, key: str, default=None):
    for holder in (blob.get("summary", {}), blob, blob.get("config", {})):
        if isinstance(holder, dict) and key in holder and holder[key] is not None:
            return holder[key]
    return default


def read_ladder(tag: str, model_id: str) -> dict:
    path = os.path.join(paths.RUNS,
                        f"ladder_{encoders.model_slug(model_id)}.json")
    if not os.path.exists(path):
        path = os.path.join(paths.RUNS, f"ladder_{tag}.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"[{NAME}] missing ladder JSON: {path}\n"
            "the steering run depends on the ladder for the chosen layer, the pooling and the trained probe.\n"
            f"Run:  python -m experiments.main_results.ladder --model {model_id}")
    with open(path) as f:
        blob = json.load(f)
    layer = _dig(blob, "best_layer", _dig(blob, "chosen_layer"))
    if layer is None:
        raise SystemExit(
            f"[{NAME}] {path} has no 'best_layer' (or 'chosen_layer') in summary / top "
            "level / config. Pass --layer to override.")
    return {"path": path, "layer": int(layer), "pooling": _dig(blob, "pooling", "last"),
            "pca_dim": _dig(blob, "pca_dim", _dig(blob, "pca", 768)),
            "pca_path": _dig(blob, "pca_path")}


def probe_ckpt_path(tag: str, layer: int, seed: int) -> str:
    return os.path.join(paths.CACHE_DEC, tag,
                        f"probe_E8_infonce_L{layer}_seed{seed}.pt")


class _PCAInverse:

    def __init__(self, components: torch.Tensor, mean: torch.Tensor):
        comp = torch.as_tensor(components).float()
        mean = torch.as_tensor(mean).float().flatten()
        D = mean.numel()
        if comp.shape[0] != D and comp.shape[1] == D:
            comp = comp.t().contiguous()
        if comp.shape[0] != D:
            raise ValueError(f"PCA components {tuple(comp.shape)} do not match mean {D}")
        self.P, self.mean = comp, mean

    @property
    def k(self) -> int:
        return int(self.P.shape[1])

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X.float() - self.mean) @ self.P

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        return Z.float() @ self.P.t() + self.mean


def wrap_pca(obj):
    if obj is None:
        return None
    if hasattr(obj, "inverse_transform") and hasattr(obj, "transform"):
        return obj
    if hasattr(obj, "components") and hasattr(obj, "mean"):
        return _PCAInverse(obj.components, obj.mean)
    if isinstance(obj, dict):
        comp = obj.get("components", obj.get("P", obj.get("V")))
        mean = obj.get("mean", obj.get("mu"))
        if comp is None or mean is None:
            raise SystemExit(f"[{NAME}] stored PCA has keys {sorted(obj)}; "
                             "'components' and 'mean' are required")
        return _PCAInverse(comp, mean)
    raise SystemExit(f"[{NAME}] cannot interpret a stored PCA of type {type(obj)}")


def refit_pca(embs_train: dict, layer: int, k: int):
    rows = torch.cat([embs_train[f][layer].float() for f in FIELDS], dim=0)
    if rows.shape[1] <= k:
        return None
    return encoders.PCAProjector.fit(rows, k=k, seed=0)


def load_probe(tag: str, layer: int, seed: int, device):
    path = probe_ckpt_path(tag, layer, seed)
    if not os.path.exists(path):
        raise SystemExit(
            f"[{NAME}] missing probe checkpoint: {path}\n"
            "the steering run re-uses the E=8 InfoNCE mixture trained by the ladder. ladder.py "
            "must save, for the mixture8_infonce row at the best layer and each seed:\n"
            "  torch.save({'state_dict': probe.state_dict(), 'E': args.experts,\n"
            "              'D': x_tr.shape[1], 'layer': best, 'seed': s,\n"
            "              'pooling': pooling, 'loss': 'infonce',\n"
            "              'pca': tensors_pca['proj'], 'gate_type': 'softmax',\n"
            "              'use_shared': False}, <this path>)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    E = int(ckpt.get("E", 8))
    D = ckpt.get("D")
    if D is None:
        w = state.get("experts")
        if w is None:
            raise SystemExit(f"[{NAME}] {path} has no 'D' and no 'experts' tensor")
        D = int(w.shape[-1])
    probe = probes.make_probe(int(D), E=E, gate_type=ckpt.get("gate_type", "softmax"),
                              use_shared=bool(ckpt.get("use_shared", False)))
    try:
        probe.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            f"[{NAME}] the checkpoint at {path} does not fit FlexMixtureOfW(D={D}, E={E}, "
            f"gate_type={ckpt.get('gate_type', 'softmax')!r}, "
            f"use_shared={bool(ckpt.get('use_shared', False))}): {exc}\n"
            "Record 'gate_type' and 'use_shared' in the checkpoint if the ladder varied them.")
    probe.to(device).eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe, ckpt, int(D), E


def stratified_sample(df, n: int, min_relations: int = 10, seed: int = 0,
                      max_relations: int = 30, prefer=()):
    g = torch.Generator().manual_seed(int(seed))
    groups = []
    for rel, sub in df.groupby("pred_value", sort=False):
        pos = [int(df.index.get_loc(i)) for i in sub.index]
        perm = torch.randperm(len(pos), generator=g).tolist()
        groups.append((rel, [pos[i] for i in perm]))
    prefer = set(prefer)
    groups.sort(key=lambda kv: (0 if kv[0] in prefer else 1, -len(kv[1])))
    if len(groups) < min_relations:
        raise SystemExit(f"[{NAME}] only {len(groups)} relations in the split, "
                         f"{min_relations} required")
    if max_relations and max_relations >= min_relations:
        groups = groups[:max_relations]
    chosen: list = []
    depth = 0
    while len(chosen) < n:
        added = 0
        for _, idxs in groups:
            if depth < len(idxs):
                chosen.append(idxs[depth])
                added += 1
                if len(chosen) >= n:
                    break
        if added == 0:
            break
        depth += 1
    chosen = sorted(chosen)
    out = df.iloc[chosen].reset_index(drop=True)
    if out["pred_value"].nunique() < min_relations:
        raise SystemExit(f"[{NAME}] sample covers {out['pred_value'].nunique()} relations, "
                         f"{min_relations} required")
    return out, chosen


def _mean_ci(x: torch.Tensor, seed: int = 0) -> dict:
    return T.paired_bootstrap(x.float(), torch.zeros_like(x.float()), n_boot=2000, seed=seed)


def monotone_steps(gains) -> int:
    seq = [0.0] + list(gains)
    return sum(1 for a, b in zip(seq, seq[1:]) if b >= a)


def verdict_from(best_gain: float, best_ci_low: float, steps: int,
                 decode_probe: float, decode_random: float) -> str:
    sig = best_ci_low > 0.0
    steer_green = (best_gain >= THRESHOLDS["gain_nats_green"] and sig
                   and steps >= THRESHOLDS["monotone_min_steps"])
    decode_green = (decode_probe >= THRESHOLDS["decode_rate_green"]
                    and decode_random <= THRESHOLDS["decode_rate_random_max"])
    if steer_green and decode_green:
        return "green"
    if (sig and best_gain > 0.0) or decode_probe >= THRESHOLDS["decode_rate_amber"]:
        return "amber"
    return "red"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="steering with the map's predicted direction")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="wikidata")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default=None, help="default: encoders.model_slug(model)")
    ap.add_argument("--layer", type=int, default=-1, help="override the ladder's layer")
    ap.add_argument("--patch-layer", type=int, default=-1,
                    help="hidden-state layer at which the patchscope arm writes into the "
                         "identity prompt. Default -1 keeps the probe's own layer. The "
                         "identity mapping needs depth below the patch to execute: "
                         "measured on Qwen3.5-4B, the h_o ceiling decodes at 0.000 when "
                         "patched at layer 26 of 32 and at 0.250 when patched at layer 3. "
                         "Choose it on the ceiling, never on the probe.")
    ap.add_argument("--pooling", default="", help="override the ladder's pooling")
    ap.add_argument("--position", default="last", choices=["last", "subject_last"],
                    help="token whose residual is steered; the default is the last "
                         "subject token, the default here is the last prompt token")
    ap.add_argument("--min-relations", type=int, default=10)
    ap.add_argument("--max-relations", type=int, default=30,
                    help="cap the number of relations so each keeps enough triples for a "
                         "same-relation foil; templated relations are preferred")
    ap.add_argument("--batch", type=int, default=16, help="prompts per forward pass")
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--refit-pca", action="store_true",
                    help="refit the label-free projector from the cached train rows when "
                         "the probe checkpoint does not carry one")
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="32 triples, 2 alphas, 1 seed")
    args = ap.parse_args(argv)
    args.alphas = sorted(float(a) for a in str(args.alphas).split(",") if a.strip())
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if args.smoke:
        args.n, args.alphas, args.seeds = 32, args.alphas[:2], args.seeds[:1]
    if args.tag is None:
        args.tag = encoders.model_slug(args.model) + ("_smoke" if args.smoke else "")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    C.set_seed(0)
    tag = args.tag
    ladder = read_ladder(tag, args.model)
    layer = args.layer if args.layer >= 0 else ladder["layer"]
    pooling = args.pooling or ladder["pooling"]
    module_layer = max(encoders.module_index(int(layer)), 0)

    run = registry.Run(NAME, tag, config=vars(args), thresholds=THRESHOLDS)
    run.set_config({"layer_hidden": int(layer), "layer_module": int(module_layer),
                    "pooling": pooling, "ladder_json": ladder["path"]})

    device = gpu_guard.acquire(min_free_gb=args.min_free_gb, allow_cpu=args.allow_cpu)

    df_full = data.load_dataset(args.dataset, "test")
    df, positions = stratified_sample(
        df_full, args.n, args.min_relations, seed=0, max_relations=args.max_relations,
        prefer=causal.DATASET_TEMPLATES.get(args.dataset, {}).keys())
    subs = df["subject"].astype(str).tolist()
    rels = df["pred_value"].astype(str).tolist()
    objs = df["object"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, args.dataset)

    tok, model = causal.load_model_and_tokenizer(args.model, device)
    gold_ids = causal.gold_token_ids(tok, objs)
    foil_ids = causal.gold_token_ids(tok, causal.pick_foils(rels, objs, seed=0))
    offsets = (causal.subject_end_offsets(tok, prompts, subs)
               if args.position == "subject_last" else None)

    sel = torch.tensor(positions, dtype=torch.long)
    if args.smoke:
        picked = {f: encoders.embed_strings(df[f].astype(str).tolist(), tok, model,
                                            [layer], pooling, batch=args.embed_batch,
                                            max_len=32)[layer].float()
                  for f in FIELDS}
    else:
        picked = {}
        for f in FIELDS:
            full = encoders.load_or_embed(df_full[f].astype(str).tolist(), tok, model,
                                          [layer], pooling, args.model, args.dataset,
                                          "test", f, batch=args.embed_batch, max_len=32)
            picked[f] = full[layer].float()[sel]
    h_s, h_r, h_o = picked["subject"], picked["pred_value"], picked["object"]

    notes = [
        "gold token = first token of ' ' + object; multi-token objects are scored by "
        "their first token only",
        f"steering position = {args.position} (the default is the last subject token)",
        f"hidden-state layer {layer} is the output of module {module_layer}; the hooks "
        "act on that module",
        "alpha is in units of the residual norm at the intervened position: the added "
        "vector is alpha * ||h_pos|| * unit(o_hat - h_s)",
        "KL is KL(intervened || baseline) in nats",
        "template coverage = %.3f; the rest use the generic fallback template"
        % causal.template_coverage(rels, args.dataset),
        "Hase et al. (2023) caveat: a dose response shows the direction is SUFFICIENT to "
        "steer the output; it does not show the unperturbed forward pass uses it",
    ]

    per_seed_steer, per_seed_decode = [], []
    rows_path = run.aux_path("rows.pt")
    rr_store = (torch.load(rows_path, map_location="cpu")
                if os.path.exists(rows_path) else {})

    for seed in args.seeds:
        probe, ckpt, D_probe, _ = load_probe(tag, layer, seed, device)
        pca = wrap_pca(ckpt.get("pca"))
        if pca is None and ckpt.get("pca_path") and os.path.exists(ckpt["pca_path"]):
            pca = wrap_pca(torch.load(ckpt["pca_path"], map_location="cpu",
                                      weights_only=False))
        if pca is None and D_probe != h_s.shape[1]:
            if not args.refit_pca:
                raise SystemExit(
                    f"[{NAME}] the probe works in {D_probe} dimensions but the embeddings "
                    f"are {h_s.shape[1]}-dimensional and the checkpoint carries no PCA. "
                    "Save 'pca' in the checkpoint (preferred) or pass --refit-pca.")
            embs_tr = {f: encoders.load_or_embed(
                data.load_dataset(args.dataset, "train")[f].astype(str).tolist(), tok,
                model, [layer], pooling, args.model, args.dataset, "train", f,
                batch=args.embed_batch, max_len=32) for f in FIELDS}
            pca = refit_pca(embs_tr, layer, args.pca)
            run.note("PCA refitted locally from the cached train rows (--refit-pca); it "
                     "matches the ladder only if the cached embeddings are identical")

        hs_p = pca.transform(h_s) if pca is not None else h_s
        hr_p = pca.transform(h_r) if pca is not None else h_r
        with torch.no_grad():
            out = probe(V.compose(hs_p.to(device), hr_p.to(device), "add"), hr_p.to(device))
            o_hat_p = (out[0] if isinstance(out, tuple) else out).float().cpu()
        o_hat = pca.inverse_transform(o_hat_p) if pca is not None else o_hat_p

        key_steer = registry.config_key({"stage": "steer", "seed": seed, "layer": layer,
                                         "n": len(df), "alphas": args.alphas,
                                         "pooling": pooling, "position": args.position,
                                         "dataset": args.dataset})
        if run.done(key_steer):
            per_seed_steer.append(run.get(key_steer))
        else:
            delta = o_hat - h_s
            dirs = {
                "probe": delta,
                "random": causal.random_direction(delta.shape[1], seed=1000 + seed,
                                                  n=delta.shape[0],
                                                  ref_norm=delta.norm(dim=-1)),
                "shuffled": causal.shuffled_directions(o_hat, seed=2000 + seed) - h_s,
            }
            base = causal.steer(model, tok, prompts, delta, [0.0], module_layer,
                                position=args.position, offsets=offsets,
                                batch_size=args.batch, device=device)[0.0]
            base_m = causal.steering_metrics(base, base, gold_ids, foil_ids)
            res = {"baseline": {k: v for k, v in base_m.items() if k != "rows"},
                   "conditions": {}, "gain_vs_random": {}, "gain_vs_shuffled": {},
                   "gain_vs_baseline": {}}
            logp_rows: dict = {}
            for cond, d in dirs.items():
                res["conditions"][cond] = {}
                logp_rows[cond] = {}
                for a in args.alphas:
                    lg = causal.steer(model, tok, prompts, d, [a], module_layer,
                                      position=args.position, offsets=offsets,
                                      batch_size=args.batch, device=device)[a]
                    m = causal.steering_metrics(lg, base, gold_ids, foil_ids)
                    logp_rows[cond][a] = m["rows"]["logp_gold"].clone()
                    rr_store[f"seed{seed}_{cond}_a{a}_rank"] = m["rows"]["rank_gold"].clone()
                    res["conditions"][cond][str(a)] = {k: v for k, v in m.items()
                                                       if k != "rows"}
                    del lg
                print(f"[{NAME}] seed {seed} condition {cond}: "
                      f"logP(gold) by alpha = "
                      + ", ".join(f"{a}:{float(logp_rows[cond][a].mean()):.3f}"
                                  for a in args.alphas))
                if gpu_guard.check_yield():
                    torch.save(rr_store, rows_path)
                    gpu_guard.yield_and_exit(f"{NAME}: steering, seed {seed}")
            for a in args.alphas:
                for ctrl, field in (("random", "gain_vs_random"),
                                    ("shuffled", "gain_vs_shuffled")):
                    res[field][str(a)] = T.paired_bootstrap(
                        logp_rows["probe"][a], logp_rows[ctrl][a], n_boot=2000, seed=seed)
                    rr_store[f"seed{seed}_gain_{ctrl}_a{a}"] = (
                        logp_rows["probe"][a] - logp_rows[ctrl][a]).clone()
                res["gain_vs_baseline"][str(a)] = _mean_ci(
                    logp_rows["probe"][a] - base_m["rows"]["logp_gold"], seed=seed)
            run.put(key_steer, res)
            per_seed_steer.append(res)

        key_dec = registry.config_key({"stage": "patchscope", "seed": seed, "layer": layer,
                                       "n": len(df), "pooling": pooling,
                                       "dataset": args.dataset,
                                       "patch_layer": int(args.patch_layer)})
        patch_layer = (module_layer if args.patch_layer < 0
                       else max(encoders.module_index(int(args.patch_layer)), 0))
        if run.done(key_dec):
            per_seed_decode.append(run.get(key_dec))
        else:
            vecs = {"o_hat": o_hat,
                    "h_o": h_o,
                    "random": causal.random_direction(h_o.shape[1], seed=3000 + seed,
                                                      n=h_o.shape[0],
                                                      ref_norm=h_o.norm(dim=-1)),
                    "h_s": h_s}
            dec = {}
            for vname, v in vecs.items():
                d = causal.patchscope_decode(model, tok, v, patch_layer,
                                             gold_ids=gold_ids, batch_size=args.batch,
                                             device=device)
                rr_store[f"seed{seed}_decode_{vname}_hit"] = d["hit"].clone()
                dec[vname] = {"decode_rate": d["decode_rate"],
                              "p_gold_mean": float(d["p_gold"].mean()),
                              "rank_gold_mean": float(d["rank_gold"].mean()),
                              "hits10": float((d["rank_gold"] <= 10).float().mean()),
                              "target_norm": d["target_norm"],
                              "ci": _mean_ci(d["hit"], seed=seed)}
                print(f"[{NAME}] patchscope seed {seed} {vname:8s} decode rate "
                      f"{d['decode_rate']:.3f}")
            run.put(key_dec, dec)
            per_seed_decode.append(dec)
        del probe
        torch.save(rr_store, rows_path)

    def avg(values):
        vals = [float(v) for v in values if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    gains = [avg([r["gain_vs_random"][str(a)]["diff"] for r in per_seed_steer])
             for a in args.alphas]
    ci_lows = [avg([r["gain_vs_random"][str(a)]["ci_low"] for r in per_seed_steer])
               for a in args.alphas]
    gains_shuf = [avg([r["gain_vs_shuffled"][str(a)]["diff"] for r in per_seed_steer])
                  for a in args.alphas]
    best_i = max(range(len(args.alphas)), key=lambda i: gains[i])
    steps = monotone_steps(gains)
    dec_rate = {k: avg([d[k]["decode_rate"] for d in per_seed_decode])
                for k in ("o_hat", "h_o", "random", "h_s")}

    verdict = verdict_from(gains[best_i], ci_lows[best_i], steps,
                           dec_rate["o_hat"], dec_rate["random"])
    summary = {
        "model": args.model, "dataset": args.dataset, "layer_hidden": int(layer),
        "layer_module": int(module_layer), "pooling": pooling,
        "n_triples": int(len(df)), "n_relations": int(df["pred_value"].nunique()),
        "alphas": args.alphas, "seeds": args.seeds, "position": args.position,
        "best_alpha": args.alphas[best_i],
        "gain_nats_vs_random_best": gains[best_i],
        "gain_ci_low_best": ci_lows[best_i],
        "gain_by_alpha_vs_random": dict(zip([str(a) for a in args.alphas], gains)),
        "gain_by_alpha_vs_shuffled": dict(zip([str(a) for a in args.alphas], gains_shuf)),
        "monotone_steps": steps, "monotone_of": len(args.alphas),
        "decode_rate_o_hat": dec_rate["o_hat"], "decode_rate_h_o": dec_rate["h_o"],
        "decode_rate_random": dec_rate["random"], "decode_rate_h_s": dec_rate["h_s"],
        "baseline_p_gold": avg([r["baseline"]["p_gold_mean"] for r in per_seed_steer]),
    }
    run.set_summary(summary)
    run.set_verdict(verdict)
    for line in notes:
        run.note(line)
    torch.save(rr_store, rows_path)

    rows = [{"what": f"steer alpha={a}", "value": round(g, 4), "ci_low": round(c, 4),
             "against": "random direction"}
            for a, g, c in zip(args.alphas, gains, ci_lows)]
    rows += [{"what": f"decode {k}", "value": round(v, 4), "ci_low": "",
              "against": "argmax == gold first token"} for k, v in dec_rate.items()]
    rows.append({"what": "VERDICT", "value": verdict, "ci_low": "",
                 "against": f"monotone {steps}/{len(args.alphas)}"})
    run.summary(rows)


if __name__ == "__main__":
    main()
