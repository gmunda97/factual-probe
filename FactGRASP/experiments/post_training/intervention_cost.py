# perplexity cost of the intervention, the base cost and post cost columns of table 12
# sentences_for_perplexity is imported by cross_model_steering

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from factgrasp import paths
from factgrasp import gpu_guard, registry, encoders
from factgrasp import causal
from factgrasp import data as D

from experiments.post_training import steering as STEER

NAME = "intervention_cost"
RUNS = os.path.join(paths.ROOT, "results", "runs")
os.makedirs(RUNS, exist_ok=True)

THRESHOLDS = {
    "patch_decode_share_green": 0.30, "patch_decode_share_amber": 0.15, "patch_decode_control_green": 0.05,
    "patch_decode_control_red": 0.10, "patch_decode_flip_rate_causal": 0.6,
    "ppl_cost_ppl_change_green": 0.05, "ppl_cost_ppl_change_amber": 0.10, "ppl_cost_alpha": 1.0,
    "decode_vs_ceiling_probe_over_ceiling_green": 0.5, "decode_vs_ceiling_probe_over_random_green": 3.0,
    "decode_vs_ceiling_probe_over_ceiling_amber": 0.25, "decode_vs_ceiling_probe_over_random_amber": 2.0,
}


def sentences_for_perplexity(n: int, seed: int = 0):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        txt = [t.strip() for t in ds["text"] if len(t.strip()) > 120]
        if len(txt) >= n:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(txt), generator=g)[:n].tolist()
            return [txt[i] for i in idx], "wikitext-2-raw-v1 test"
    except Exception as exc:
        print(f"[{NAME}] wikitext unavailable ({type(exc).__name__}); using LRE prompts",
              flush=True)
    df = D.load_dataset("lre", "test")
    subs, rels = df["subject"].astype(str).tolist(), df["pred_value"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, "lre")
    objs = df["object"].astype(str).tolist()
    txt = [f"{p} {o}." for p, o in zip(prompts, objs)]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(txt), generator=g)[:n].tolist()
    return [txt[i] for i in idx], "LRE prompts with their gold objects"


def main() -> int:
    ap = argparse.ArgumentParser(description="perplexity cost of steering with the map's direction")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="wikidata")
    ap.add_argument("--parts", default="capability,containment,lre_edit")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-sentences", type=int, default=200)
    ap.add_argument("--n-lre", type=int, default=400)
    ap.add_argument("--patch-layer", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    tag = args.tag or (encoders.model_slug(args.model) + ("_smoke" if args.smoke else ""))
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if args.smoke:
        args.n, args.n_sentences, args.n_lre = 24, 16, 24
        seeds = seeds[:1]

    lad = STEER.read_ladder(encoders.model_slug(args.model), args.model)
    layer = int(lad["layer"])
    pooling = str(lad["pooling"])
    module_layer = max(encoders.module_index(layer), 0)
    patch_module = max(encoders.module_index(args.patch_layer), 0)

    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    run = registry.Run(NAME, tag, thresholds=THRESHOLDS, out_dir=RUNS)
    run.set_config(dict(vars(args), layer_hidden=layer, layer_module=module_layer,
                        patch_module=patch_module, pooling=pooling, device=str(device)))

    tok, model = causal.load_model_and_tokenizer(args.model, device)
    print(f"[{NAME}] {args.model}: probe layer {layer} (module {module_layer}), patch at "
          f"hidden layer {args.patch_layer} (module {patch_module}), pooling {pooling}",
          flush=True)

    df, _chosen = STEER.stratified_sample(D.load_dataset(args.dataset, "test"), args.n,
                                       min_relations=2 if args.smoke else 10, seed=0)
    subs = df["subject"].astype(str).tolist()
    rels = df["pred_value"].astype(str).tolist()
    objs = df["object"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, args.dataset)
    gold_ids = causal.gold_token_ids(tok, objs)

    embs = {f: encoders.load_or_embed(df[f].astype(str).tolist(), tok, model, [layer],
                                      pooling, model_id=args.model, dataset=args.dataset,
                                      split=f"capability_n{len(df)}", field=f,
                                      batch=args.embed_batch, device=device,
                                      progress=False)[layer].float()
            for f in ("subject", "pred_value", "object")}
    h_s, h_r, h_o = embs["subject"], embs["pred_value"], embs["object"]

    probe, ckpt, D_probe, E_probe = STEER.load_probe(encoders.model_slug(args.model), layer,
                                                  seeds[0], device)
    pca = STEER.wrap_pca(ckpt.get("pca"))
    with torch.no_grad():
        x = (h_s + h_r)
        xp = pca.transform(x) if pca is not None else x
        out = probe(xp.to(device), (pca.transform(h_r) if pca is not None else h_r).to(device))
        o_hat_p = (out[0] if isinstance(out, tuple) else out).float().cpu()
    o_hat = pca.inverse_transform(o_hat_p) if pca is not None else o_hat_p

    summary, measured, verdicts = {"model": args.model, "layer_hidden": layer,
                                   "layer_module": module_layer, "pooling": pooling,
                                   "n_triples": int(len(df))}, {}, {}

    if "capability" in parts:
        key = registry.config_key({"part": "capability", "alpha": args.alpha,
                                   "n": args.n_sentences, "smoke": args.smoke})
        if run.done(key):
            cap = run.get(key)
        else:
            sents, src = sentences_for_perplexity(args.n_sentences)
            base = causal.perplexity(model, tok, sents, device=device)
            delta = (o_hat - h_s).mean(0)
            one = causal.perplexity(model, tok, sents, layer_idx=module_layer,
                                    direction=delta, alpha=args.alpha, device=device,
                                    positions="first")
            allp = causal.perplexity(model, tok, sents, layer_idx=module_layer,
                                     direction=delta, alpha=args.alpha, device=device,
                                     positions="all")
            rel = (one["perplexity"] - base["perplexity"]) / max(base["perplexity"], 1e-9)
            rel_all = (allp["perplexity"] - base["perplexity"]) / max(base["perplexity"], 1e-9)
            cap = {"source": src, "n_sentences": len(sents), "alpha": args.alpha,
                   "positions_scored": "first (the dose the experiment applies)",
                   "perplexity_base": base["perplexity"], "nll_base": base["nll"],
                   "perplexity_steered": one["perplexity"], "nll_steered": one["nll"],
                   "relative_change": rel,
                   "perplexity_steered_all_positions": allp["perplexity"],
                   "relative_change_all_positions": rel_all,
                   "n_tokens": base["n_tokens"]}
            run.put(key, cap)
        summary["capability"] = cap
        rel = cap["relative_change"]
        verdicts["ppl_cost"] = ("green" if abs(rel) <= THRESHOLDS["ppl_cost_ppl_change_green"]
                          else "amber" if abs(rel) <= THRESHOLDS["ppl_cost_ppl_change_amber"]
                          else "red")
        measured["ppl_cost"] = (f"perplexity {cap['perplexity_base']:.2f} without steering and "
                          f"{cap['perplexity_steered']:.2f} at alpha {args.alpha} injected "
                          f"at one position, relative change {100 * rel:+.2f} percent on "
                          f"{cap['n_sentences']} sentences from {cap['source']}; injecting "
                          f"at every position gives "
                          f"{100 * cap['relative_change_all_positions']:+.1f} percent")
        print(f"[{NAME}] {measured['ppl_cost']}", flush=True)

    if "containment" in parts:
        key = registry.config_key({"part": "containment", "patch_layer": args.patch_layer,
                                   "n": int(len(df)), "smoke": args.smoke})
        if run.done(key):
            con = run.get(key)
        else:
            rnd = causal.random_direction(h_o.shape[1], seed=3000, n=h_o.shape[0],
                                          ref_norm=h_o.norm(dim=-1))
            vecs = {"o_hat": o_hat, "h_o": h_o, "random": rnd, "h_s": h_s}
            con = {}
            for name, v in vecs.items():
                r = causal.patchscope_contains(model, tok, v, patch_module, objs,
                                               max_new_tokens=args.max_new_tokens,
                                               batch_size=args.batch, device=device)
                con[name] = {"contains_rate": r["contains_rate"],
                             "example_generations": r["generations"][:5]}
                print(f"[{NAME}] containment {name:7s} {r['contains_rate']:.4f}", flush=True)
            run.put(key, con)
        summary["containment"] = con
        p, c, rr = (con["o_hat"]["contains_rate"], con["h_o"]["contains_rate"],
                    con["random"]["contains_rate"])
        ratio_c = p / c if c > 0 else float("nan")
        ratio_r = p / rr if rr > 0 else float("inf")
        verdicts["decode_vs_ceiling"] = ("green" if ratio_c >= THRESHOLDS["decode_vs_ceiling_probe_over_ceiling_green"]
                          and ratio_r >= THRESHOLDS["decode_vs_ceiling_probe_over_random_green"]
                          else "amber" if ratio_c >= THRESHOLDS["decode_vs_ceiling_probe_over_ceiling_amber"]
                          and ratio_r >= THRESHOLDS["decode_vs_ceiling_probe_over_random_amber"] else "red")
        measured["decode_vs_ceiling"] = (f"probe {p:.4f}, ceiling {c:.4f}, random {rr:.4f}; probe is "
                          f"{ratio_c:.2f} of the ceiling and {ratio_r:.2f} times random")

    if "lre_edit" in parts:
        key = registry.config_key({"part": "lre_edit", "n": args.n_lre,
                                   "smoke": args.smoke})
        if run.done(key):
            lre = run.get(key)
        else:
            lre = lre_edit(model, tok, probe, pca, args, device, layer, module_layer)
            run.put(key, lre)
        summary["lre_edit"] = lre
        if not lre.get("intervention_lands", True):
            measured["patch_decode"] = ("not measured: the edit does not reach the output, mean "
                              f"absolute logit shift "
                              f"{lre.get('mean_abs_logit_shift_probe', float('nan')):.5f}")
            verdicts["patch_decode"] = "instrument failure"
        elif "share_causal" in lre:
            share, ctrl = lre["share_causal"], lre["share_causal_control"]
            verdicts["patch_decode"] = ("green" if share >= THRESHOLDS["patch_decode_share_green"]
                              and ctrl <= THRESHOLDS["patch_decode_control_green"]
                              else "red" if share < THRESHOLDS["patch_decode_share_amber"]
                              or ctrl > THRESHOLDS["patch_decode_control_red"] else "amber")
            measured["patch_decode"] = (f"{100 * share:.1f} percent of {lre['n_relations']} relations "
                              f"flip at 0.6 or more, control {100 * ctrl:.1f} percent; mean "
                              f"flip rate {lre['mean_flip_probe']:.3f} against "
                              f"{lre['mean_flip_control']:.3f} and "
                              f"{lre['mean_flip_unedited']:.3f} unedited; the edit reaches "
                              f"the output, mean absolute logit shift "
                              f"{lre['mean_abs_logit_shift_probe']:.3f}")
            print(f"[{NAME}] {measured['patch_decode']}", flush=True)

    p5 = os.path.join(paths.RUNS,
                      f"steering_{encoders.model_slug(args.model)}.json")
    if os.path.exists(p5):
        steer_summary = json.load(open(p5)).get("summary", {})
        g, lo = steer_summary.get("gain_nats_vs_random_best"), steer_summary.get("gain_ci_low_best")
        if g is not None:
            steps, of = steer_summary.get("monotone_steps"), steer_summary.get("monotone_of")
            label = "steering_gain" if "gemma" in args.model.lower() else "steering_first_family"
            verdicts[label] = ("green" if g >= 0.5 and (lo or 0) > 0
                               and steps >= max(3, of - 1)
                               else "amber" if g >= 0.25 else "red")
            measured[label] = (f"steering gain {g:.3f} nats at best alpha, interval from "
                               f"{lo:.3f}, monotone {steps} of {of}, on {args.model}")
        summary["steering_from_steering"] = {k: v for k, v in steer_summary.items()
                                         if k.startswith(("gain", "monotone", "decode"))}

    summary["verdicts"] = verdicts
    summary["measured"] = measured
    run.set_summary(summary)
    judged = [v for v in verdicts.values() if v in ("red", "amber", "green")]
    run.set_verdict(min(judged, key=lambda x: ["red", "amber", "green"].index(x))
                    if judged else "n/a")
    run.note("Steering itself is unchanged from the steering run; steering_gain is read from that script run on "
             "this model. The capability check steers with the mean probe direction at "
             "every position of an unrelated sentence, which is the harsher reading.")

    srows = []
    for k in ("patch_decode", "ppl_cost", "decode_vs_ceiling", "steering_gain"):
        if k in measured:
            srows.append({"hypothesis": k, "verdict": verdicts.get(k, "-"),
                          "measured": measured[k][:80]})
    run.summary(srows)
    return 0


def lre_edit(model, tok, probe, pca, args, device, layer, module_layer) -> dict:
    df = D.load_dataset("lre", "test")
    g = torch.Generator().manual_seed(0)
    if args.n_lre and args.n_lre < len(df):
        df = df.iloc[sorted(torch.randperm(len(df), generator=g)[:args.n_lre].tolist())]
    df = df.reset_index(drop=True)
    subs = df["subject"].astype(str).tolist()
    rels = df["pred_value"].astype(str).tolist()
    objs = df["object"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, "lre")
    offsets = causal.subject_end_offsets(tok, prompts, subs)

    by_rel = {}
    for i, r in enumerate(rels):
        by_rel.setdefault(r, []).append(i)
    targets = []
    for i, r in enumerate(rels):
        pool = [j for j in by_rel[r] if objs[j] != objs[i]]
        targets.append(objs[pool[(i * 7) % len(pool)]] if pool else objs[i])
    target_ids = causal.gold_token_ids(tok, targets)
    gold_ids = causal.gold_token_ids(tok, objs)

    embs = {f: encoders.load_or_embed(df[c].astype(str).tolist(), tok, model, [layer],
                                      "last", model_id=args.model, dataset="lre",
                                      split=f"capability{len(df)}", field=f,
                                      batch=args.embed_batch, device=device,
                                      progress=False)[layer].float()
            for f, c in (("subject", "subject"), ("pred_value", "pred_value"))}
    h_t = encoders.load_or_embed(targets, tok, model, [layer], "last",
                                 model_id=args.model, dataset="lre",
                                 split=f"capability_targets{len(df)}", field="object",
                                 batch=args.embed_batch, device=device,
                                 progress=False)[layer].float()
    h_s, h_r = embs["subject"], embs["pred_value"]
    hs_p = pca.transform(h_s) if pca is not None else h_s
    hr_p = pca.transform(h_r) if pca is not None else h_r
    ht_p = pca.transform(h_t) if pca is not None else h_t

    with torch.no_grad():
        gate = probe.gate_weights(hr_p.to(device)).cpu()
        experts = probe.experts.detach().cpu().float()
    s_new = torch.empty_like(hs_p)
    for i in range(hs_p.shape[0]):
        W = torch.einsum("e,edk->dk", gate[i], experts)
        sol = torch.linalg.lstsq(W, ht_p[i].unsqueeze(1)).solution.squeeze(1)
        s_new[i] = sol - hr_p[i]
    s_model = pca.inverse_transform(s_new) if pca is not None else s_new
    s_model = causal.rescale_to_norm(s_model, h_s.norm(dim=-1))
    ctrl = causal.random_direction(h_s.shape[1], seed=4000, n=h_s.shape[0],
                                   ref_norm=h_s.norm(dim=-1))

    probe_r = causal.subject_edit_flip(model, tok, prompts, s_model, module_layer, offsets,
                                       target_ids, gold_ids, batch_size=args.batch,
                                       device=device)
    ctrl_r = causal.subject_edit_flip(model, tok, prompts, ctrl, module_layer, offsets,
                                      target_ids, gold_ids, batch_size=args.batch,
                                      device=device)
    base_r = causal.subject_edit_flip(model, tok, prompts, None, module_layer, offsets,
                                      target_ids, gold_ids, batch_size=args.batch,
                                      device=device)
    def _logit_shift(a, b):
        if a.get("logits") is None or b.get("logits") is None:
            return float("nan")
        return float((a["logits"] - b["logits"]).abs().mean())
    shift_probe = _logit_shift(probe_r, base_r)
    shift_ctrl = _logit_shift(ctrl_r, base_r)
    lands = bool(shift_probe > 1e-3)

    per_rel, rows = {}, []
    for i, r in enumerate(rels):
        per_rel.setdefault(r, {"p": [], "c": []})
        per_rel[r]["p"].append(float(probe_r["flip"][i]))
        per_rel[r]["c"].append(float(ctrl_r["flip"][i]))
    for r, v in sorted(per_rel.items()):
        if len(v["p"]) < 3:
            continue
        rows.append({"relation": r, "n_prompts": len(v["p"]),
                     "flip_rate_probe": sum(v["p"]) / len(v["p"]),
                     "flip_rate_random": sum(v["c"]) / len(v["c"])})
    thr = THRESHOLDS["patch_decode_flip_rate_causal"]
    n = len(rows) or 1
    return {"n_relations": len(rows), "n_prompts": int(len(df)),
            "share_causal": sum(1 for r in rows if r["flip_rate_probe"] >= thr) / n,
            "share_causal_control": sum(1 for r in rows
                                        if r["flip_rate_random"] >= thr) / n,
            "mean_flip_probe": probe_r["flip_rate"],
            "mean_flip_control": ctrl_r["flip_rate"],
            "mean_flip_unedited": base_r["flip_rate"],
            "still_gold_probe": probe_r["still_gold_rate"],
            "still_gold_control": ctrl_r["still_gold_rate"],
            "still_gold_unedited": base_r["still_gold_rate"],
            "intervention_lands": lands,
            "mean_abs_logit_shift_probe": shift_probe,
            "mean_abs_logit_shift_control": shift_ctrl,
            "per_relation": rows}


if __name__ == "__main__":
    sys.exit(main())
