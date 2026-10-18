# steering through the alignment, source of the 11.134 nats gemma-4-12b number in appendix e

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from factgrasp import (causal, crossmodel as X, data as D, encoders as E, gpu_guard,
                  paths, registry)
from factgrasp import theory_reference as T
from experiments.post_training.steering import (FIELDS, load_probe, monotone_steps,
                                              read_ladder, stratified_sample, wrap_pca)
from experiments.post_training.intervention_cost import sentences_for_perplexity

sys.path.insert(0, paths.KG_CODE)
import variants as V

NAME = "cross_model_steering"
RUNS = os.path.join(paths.ROOT, "results", "runs")
PAIRS = [
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen3.5", "forward"),
    ("google/gemma-4-E4B", "google/gemma-4-12B", "Gemma-4", "forward"),
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B", "Llama", "forward"),
    ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen3.5", "reverse"),
    ("google/gemma-4-12B", "google/gemma-4-E4B", "Gemma-4", "reverse"),
    ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B", "Llama", "reverse"),
]
DIRECTIONS = ("native", "transferred", "ceiling", "random", "derangement")
THRESHOLDS = {
    "steer_transfer": "the transferred direction steers the target: gain over random at least 0.5 nats "
           "and monotone over at least 3 of 4 alphas, derangement below random, on 2 of 3 "
           "forward pairs (green); 0.25 to 0.5 (amber); below 0.25 on 2 of 3 (red)",
    "steer_native_ratio": "it is close to the native direction: transferred gain at least 0.7 of the native "
           "gain on 2 of 3 forward pairs (green); 0.4 to 0.7 (amber); below 0.4 on 2 of 3 (red)",
    "steer_strong_to_weak": "a stronger model's direction steers a weaker model at least as well as its own: "
           "transferred gain at least the native gain on 2 of 3 forward pairs (green); within "
           "0.1 nats on 2 of 3 (amber); below on 2 of 3 (red)",
    "steer_asymmetry": "steering transfer is asymmetric the way readout transfer is: forward-pair ratio "
           "(transferred over native) exceeds the reverse-pair ratio on 3 of 3 (green); 2 of 3 "
           "(amber); otherwise (red)",
    "steer_ceiling": "the stitch carries the object geometry: the ceiling (transferred gold vector) "
           "steers at least as well as the transferred prediction on 3 of 3 (green); 2 of 3 "
           "(amber); otherwise (red, the stitch, not the probe, is the bottleneck)",
}


def slug(m: str) -> str:
    return E.model_slug(m)


def cached_fields(model_id: str, split: str, layer: int, pooling: str, texts_by_field: dict,
                  device, tok_model: list) -> dict:
    out = {}
    for f in FIELDS:
        p = E.cache_path(model_id, "wikidata", split, f, pooling, layer)
        if not os.path.exists(p) and tok_model[0] is None:
            print(f"[{NAME}] {model_id}: cache missing for {split}/{f} at layer {layer}, "
                  f"loading weights", flush=True)
            tok_model[0], tok_model[1] = E.load_decoder(model_id, device)
        out[f] = E.load_or_embed(texts_by_field[f], tok_model[0], tok_model[1], [layer],
                                 pooling, model_id=model_id, dataset="wikidata", split=split,
                                 field=f, device=device, progress=False)[layer].float()
    return out


def stitch_map(A: str, B: str, lad_a: dict, lad_b: dict, pca_a, pca_b, device,
               ridge: float) -> torch.Tensor:
    path = os.path.join(RUNS, f"stitch_{slug(A)}__{slug(B)}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")["S"].float()
    tr = D.load_dataset("wikidata", "train")
    texts = {f: tr[f].astype(str).tolist() for f in FIELDS}
    xs = {}
    for m, lad, pca in ((A, lad_a, pca_a), (B, lad_b, pca_b)):
        tm = [None, None]
        emb = cached_fields(m, "train", lad["layer"], lad["pooling"], texts, device, tm)
        if tm[1] is not None:
            del tm[1]; torch.cuda.empty_cache()
        xs[m] = (pca.transform(emb["subject"]) + pca.transform(emb["pred_value"])).float()
    S = X.fit_stitch(xs[B], xs[A], ridge=ridge)
    torch.save({"S": S, "A": A, "B": B, "ridge": ridge, "layer_A": lad_a["layer"],
                "layer_B": lad_b["layer"], "n_train": int(xs[A].shape[0])}, path)
    print(f"[{NAME}] stitch {A} <- {B} fitted on {xs[A].shape[0]} training queries and saved",
          flush=True)
    return S


def run_pair(A: str, B: str, fam: str, kind: str, args, run: registry.Run, device) -> dict:
    t0 = time.time()
    lad_a, lad_b = read_ladder(slug(A), A), read_ladder(slug(B), B)
    layer_a, layer_b = lad_a["layer"], lad_b["layer"]
    module_a = max(E.module_index(layer_a), 0)
    key_pair = registry.config_key({"pair": f"{A} <- {B}", "n": args.n, "alphas": args.alphas,
                                    "seeds": args.seeds, "smoke": args.smoke})
    if run.done(key_pair):
        print(f"[{NAME}] {A} <- {B}: done earlier", flush=True)
        return run.get(key_pair)

    df_full = D.load_dataset("wikidata", "test")
    df, positions = stratified_sample(df_full, args.n, args.min_relations, seed=0,
                                      max_relations=args.max_relations,
                                      prefer=causal.DATASET_TEMPLATES.get("wikidata", {}).keys())
    subs = df["subject"].astype(str).tolist()
    rels = df["pred_value"].astype(str).tolist()
    objs = df["object"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, "wikidata")
    sel = torch.tensor(positions, dtype=torch.long)

    tok, model = causal.load_model_and_tokenizer(A, device)
    gold_ids = causal.gold_token_ids(tok, objs)
    foil_ids = causal.gold_token_ids(tok, causal.pick_foils(rels, objs, seed=0))
    texts = {f: df_full[f].astype(str).tolist() for f in FIELDS}
    emb_a = cached_fields(A, "test", layer_a, lad_a["pooling"], texts, device, [tok, model])
    h_s_a, h_r_a, h_o_a = (emb_a[f][sel] for f in FIELDS)

    tm_b = [None, None]
    emb_b = cached_fields(B, "test", layer_b, lad_b["pooling"], texts, device, tm_b)
    if tm_b[1] is not None:
        del tm_b[1]; torch.cuda.empty_cache()
    h_s_b, h_r_b, h_o_b = (emb_b[f][sel] for f in FIELDS)

    ck_a = torch.load(os.path.join(paths.CACHE_DEC, slug(A), f"probe_E8_infonce_L{layer_a}_seed0.pt"),
                      map_location="cpu", weights_only=False)
    ck_b = torch.load(os.path.join(paths.CACHE_DEC, slug(B), f"probe_E8_infonce_L{layer_b}_seed0.pt"),
                      map_location="cpu", weights_only=False)
    pca_a, pca_b = wrap_pca(ck_a.get("pca")), wrap_pca(ck_b.get("pca"))
    if pca_a is None or pca_b is None:
        raise SystemExit(f"[{NAME}] {A} or {B}: the E8 checkpoint carries no PCA")
    S = stitch_map(A, B, lad_a, lad_b, pca_a, pca_b, device, args.ridge)
    to_a_space = lambda z: z.float() @ pca_a.P.t()

    base = causal.steer(model, tok, prompts, torch.zeros(len(prompts), h_s_a.shape[1]), [0.0],
                        module_a, position="last", batch_size=args.batch, device=device)[0.0]
    base_m = causal.steering_metrics(base, base, gold_ids, foil_ids)

    per_seed = []
    for seed in args.seeds:
        key_seed = registry.config_key({"pair": f"{A} <- {B}", "seed": seed, "n": args.n,
                                        "alphas": args.alphas, "smoke": args.smoke})
        if run.done(key_seed):
            per_seed.append(run.get(key_seed)); continue
        probe_a, _, _, _ = load_probe(slug(A), layer_a, seed, device)
        probe_b, _, _, _ = load_probe(slug(B), layer_b, seed, device)
        with torch.no_grad():
            xa = V.compose(pca_a.transform(h_s_a), pca_a.transform(h_r_a), "add").to(device)
            out = probe_a(xa, pca_a.transform(h_r_a).to(device))
            o_hat_a_p = (out[0] if isinstance(out, tuple) else out).float().cpu()
            xb = V.compose(pca_b.transform(h_s_b), pca_b.transform(h_r_b), "add").to(device)
            out = probe_b(xb, pca_b.transform(h_r_b).to(device))
            o_hat_b_p = (out[0] if isinstance(out, tuple) else out).float().cpu()
        native = pca_a.inverse_transform(o_hat_a_p) - h_s_a
        d_b = o_hat_b_p - pca_b.transform(h_s_b)
        transferred = to_a_space(d_b @ S)
        ceiling = to_a_space((pca_b.transform(h_o_b) - pca_b.transform(h_s_b)) @ S)
        dirs = {"native": native, "transferred": transferred, "ceiling": ceiling,
                "random": causal.random_direction(native.shape[1], seed=1000 + seed,
                                                  n=native.shape[0], ref_norm=native.norm(dim=-1)),
                "derangement": causal.shuffled_directions(transferred, seed=2000 + seed)}
        cos_tn = float(torch.nn.functional.cosine_similarity(transferred, native, dim=-1).mean())
        cos_cn = float(torch.nn.functional.cosine_similarity(ceiling, native, dim=-1).mean())
        logp = {}
        res = {"A": A, "B": B, "kind": kind, "seed": seed, "cos_transferred_native": cos_tn,
               "cos_ceiling_native": cos_cn, "logp_gold_mean": {}, "gain_vs_random": {},
               "gain_vs_baseline": {}}
        for name, d in dirs.items():
            logp[name] = {}
            res["logp_gold_mean"][name] = {}
            for a in args.alphas:
                lg = causal.steer(model, tok, prompts, d, [a], module_a, position="last",
                                  batch_size=args.batch, device=device)[a]
                m = causal.steering_metrics(lg, base, gold_ids, foil_ids)
                logp[name][a] = m["rows"]["logp_gold"].clone()
                res["logp_gold_mean"][name][str(a)] = float(logp[name][a].mean())
                del lg
            print(f"[{NAME}] {A} <- {B} seed {seed} {name:12s} logP(gold) by alpha = "
                  + ", ".join(f"{a}:{res['logp_gold_mean'][name][str(a)]:.3f}" for a in args.alphas),
                  flush=True)
            if gpu_guard.check_yield():
                gpu_guard.yield_and_exit(f"{NAME}: {A} <- {B} seed {seed}")
        for name in ("native", "transferred", "ceiling", "derangement"):
            res["gain_vs_random"][name] = {
                str(a): T.paired_bootstrap(logp[name][a], logp["random"][a], n_boot=2000, seed=seed)
                for a in args.alphas}
            res["gain_vs_baseline"][name] = {
                str(a): float((logp[name][a] - base_m["rows"]["logp_gold"]).mean())
                for a in args.alphas}
        if seed == args.seeds[0]:
            sents, src = sentences_for_perplexity(args.n_sentences)
            ppl0 = causal.perplexity(model, tok, sents, device=device)
            res["capability"] = {"source": src, "n_sentences": len(sents),
                                 "perplexity_base": ppl0["perplexity"], "ppl_change": {}}
            for name in ("native", "transferred"):
                p1 = causal.perplexity(model, tok, sents, layer_idx=module_a,
                                       direction=dirs[name].mean(0), alpha=1.0, device=device,
                                       positions="first")
                res["capability"]["ppl_change"][name] = (
                    (p1["perplexity"] - ppl0["perplexity"]) / max(ppl0["perplexity"], 1e-9))
        run.put(key_seed, res); per_seed.append(res)
        del probe_a, probe_b
    del model; torch.cuda.empty_cache()

    def avg(vals):
        v = [float(x) for x in vals if x is not None]
        return sum(v) / len(v) if v else float("nan")
    summ = {"A": A, "B": B, "family": fam, "kind": kind, "layer_A": layer_a, "layer_B": layer_b,
            "n_triples": int(len(df)), "n_relations": int(df["pred_value"].nunique()),
            "alphas": args.alphas, "seeds": args.seeds,
            "baseline_logp_gold": float(base_m["logp_gold_mean"]),
            "cos_transferred_native": avg(r["cos_transferred_native"] for r in per_seed),
            "cos_ceiling_native": avg(r["cos_ceiling_native"] for r in per_seed),
            "gain_by_alpha": {}, "ci_low_by_alpha": {}, "best_alpha": {}, "best_gain": {},
            "best_ci_low": {}, "monotone_steps": {}}
    for name in ("native", "transferred", "ceiling", "derangement"):
        gains = [avg(r["gain_vs_random"][name][str(a)]["diff"] for r in per_seed) for a in args.alphas]
        lows = [avg(r["gain_vs_random"][name][str(a)]["ci_low"] for r in per_seed) for a in args.alphas]
        bi = max(range(len(args.alphas)), key=lambda i: gains[i])
        summ["gain_by_alpha"][name] = dict(zip([str(a) for a in args.alphas], gains))
        summ["ci_low_by_alpha"][name] = dict(zip([str(a) for a in args.alphas], lows))
        summ["best_alpha"][name] = args.alphas[bi]
        summ["best_gain"][name] = gains[bi]
        summ["best_ci_low"][name] = lows[bi]
        summ["monotone_steps"][name] = monotone_steps(gains)
    ta = str(summ["best_alpha"]["transferred"])
    summ["derangement_gain_at_transferred_best_alpha"] = summ["gain_by_alpha"]["derangement"][ta]
    gn = summ["best_gain"]["native"]
    summ["ratio_transferred_over_native"] = (summ["best_gain"]["transferred"] / gn
                                             if gn > 0 else float("nan"))
    cap = next((r["capability"] for r in per_seed if "capability" in r), None)
    summ["ppl_change_native"] = cap["ppl_change"]["native"] if cap else None
    summ["ppl_change_transferred"] = cap["ppl_change"]["transferred"] if cap else None
    summ["perplexity_base"] = cap["perplexity_base"] if cap else None
    summ["sentences"] = cap["source"] if cap else None
    summ["seconds"] = round(time.time() - t0)
    run.put(key_pair, summ)
    print(f"=== SUMMARY {NAME} {A} <- {B} ({kind}) ===", flush=True)
    for name in ("native", "transferred", "ceiling", "derangement"):
        print(f"  {name:12s} best alpha {summ['best_alpha'][name]:<4} gain vs random "
              f"{summ['best_gain'][name]:+.3f} (ci low {summ['best_ci_low'][name]:+.3f}), monotone "
              f"{summ['monotone_steps'][name]}/{len(args.alphas)}", flush=True)
    print(f"  ratio transferred/native {summ['ratio_transferred_over_native']:.3f}; derangement at the "
          f"transferred best alpha {summ['derangement_gain_at_transferred_best_alpha']:+.3f}; cos(transferred, "
          f"native) {summ['cos_transferred_native']:.3f}; ppl change at alpha 1 native "
          f"{(summ['ppl_change_native'] or 0) * 100:+.1f}%, transferred {(summ['ppl_change_transferred'] or 0) * 100:+.1f}%",
          flush=True)
    print("=== END ===", flush=True)
    return summ


def collect(run: registry.Run, csv_dir: str) -> int:
    rows = [v for v in run.rows().values() if isinstance(v, dict) and "ratio_transferred_over_native" in v]
    by = {(r["A"], r["B"]): r for r in rows}
    fwd = [by.get((a, b)) for a, b, f, k in PAIRS if k == "forward"]
    rev = {f: by.get((a, b)) for a, b, f, k in PAIRS if k == "reverse"}
    fams = [f for a, b, f, k in PAIRS if k == "forward"]
    have = [r for r in fwd if r]
    pending = len(have) < 3 or any(rev[f] is None for f in fams)

    def g(r, name):
        return r["best_gain"][name]
    a_green = sum(1 for r in have if g(r, "transferred") >= 0.5 and r["monotone_steps"]["transferred"] >= 3
                  and r["derangement_gain_at_transferred_best_alpha"] < 0)
    a_low = sum(1 for r in have if g(r, "transferred") < 0.25)
    v_a = "green" if a_green >= 2 else "red" if a_low >= 2 else "amber"
    ratios = [r["ratio_transferred_over_native"] for r in have]
    v_b = ("green" if sum(1 for x in ratios if x >= 0.7) >= 2 else
           "red" if sum(1 for x in ratios if x < 0.4) >= 2 else "amber")
    c_ge = sum(1 for r in have if g(r, "transferred") >= g(r, "native"))
    c_near = sum(1 for r in have if g(r, "transferred") >= g(r, "native") - 0.1)
    v_c = "green" if c_ge >= 2 else "amber" if c_near >= 2 else "red"
    d_ok = sum(1 for f, r in zip(fams, fwd) if r and rev[f] and
               r["ratio_transferred_over_native"] > rev[f]["ratio_transferred_over_native"])
    v_d = "green" if d_ok == 3 else "amber" if d_ok == 2 else "red"
    e_ok = sum(1 for r in have if g(r, "ceiling") >= g(r, "transferred"))
    e_rev = sum(1 for r in rev.values() if r and g(r, "ceiling") >= g(r, "transferred"))
    v_e = "green" if e_ok == 3 else "amber" if e_ok == 2 else "red"
    verd = {k: ("pending" if pending else v) for k, v in
            (("steer_transfer", v_a), ("steer_native_ratio", v_b), ("steer_strong_to_weak", v_c), ("steer_asymmetry", v_d), ("steer_ceiling", v_e))}
    measured = {
        "steer_transfer_forward_pairs_green": f"{a_green} of {len(have)}",
        "steer_transfer_transferred_best_gain": {f"{r['A']} <- {r['B']}": round(g(r, 'transferred'), 3) for r in have},
        "steer_native_ratio_ratio_transferred_over_native": {f"{r['A']} <- {r['B']}": round(r["ratio_transferred_over_native"], 3) for r in have},
        "steer_strong_to_weak_transferred_at_least_native": f"{c_ge} of {len(have)}; within 0.1: {c_near} of {len(have)}",
        "steer_asymmetry_forward_ratio_exceeds_reverse": f"{d_ok} of 3",
        "steer_asymmetry_reverse_ratios": {f"{r['A']} <- {r['B']}": round(r["ratio_transferred_over_native"], 3)
                               for r in rev.values() if r},
        "steer_ceiling_ceiling_at_least_transferred_forward": f"{e_ok} of {len(have)}",
        "steer_ceiling_ceiling_at_least_transferred_reverse": f"{e_rev} of {sum(1 for r in rev.values() if r)}",
        "native_best_gain": {f"{r['A']} <- {r['B']}": round(g(r, 'native'), 3) for r in rows},
        "ceiling_best_gain": {f"{r['A']} <- {r['B']}": round(g(r, 'ceiling'), 3) for r in rows},
        "ppl_change_at_alpha_1": {f"{r['A']} <- {r['B']}": {"native": r["ppl_change_native"],
                                                            "transferred": r["ppl_change_transferred"]} for r in rows},
    }
    run.set_summary({"n_pairs": len(rows), "verdicts": verd, "measured": measured})
    run.flush()
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, "cross_model_steering.csv"), "w") as fh:
        fh.write("target,source,kind,direction,alpha,seed,logp_gold,gain_vs_random,lo,hi,ppl_change\n")
        seed_rows = [v for v in run.rows().values()
                     if isinstance(v, dict) and "gain_vs_random" in v and "seed" in v]
        for v in sorted(seed_rows, key=lambda v: (v["A"], v["B"], v["seed"])):
            cap = v.get("capability", {}).get("ppl_change", {})
            for name in DIRECTIONS:
                for a, lp in v["logp_gold_mean"][name].items():
                    if name == "random":
                        gain = lo = hi = ""
                    else:
                        b = v["gain_vs_random"][name][a]
                        gain, lo, hi = f"{b['diff']:.6f}", f"{b['ci_low']:.6f}", f"{b['ci_high']:.6f}"
                    ppl = (f"{cap[name]:.6f}" if float(a) == 1.0 and name in cap else "")
                    fh.write(f"{v['A']},{v['B']},{v['kind']},{name},{a},{v['seed']},{lp:.6f},"
                             f"{gain},{lo},{hi},{ppl}\n")
        for r in rows:
            for name in ("native", "transferred", "ceiling", "derangement"):
                for a in r["alphas"]:
                    ppl = ""
                    if float(a) == 1.0 and name in ("native", "transferred") and r.get(f"ppl_change_{name}") is not None:
                        ppl = f"{r[f'ppl_change_{name}']:.6f}"
                    fh.write(f"{r['A']},{r['B']},{r['kind']},{name},{a},mean,,"
                             f"{r['gain_by_alpha'][name][str(a)]:.6f},{r['ci_low_by_alpha'][name][str(a)]:.6f},,{ppl}\n")
    print(f"=== SUMMARY {NAME} collect ===", flush=True)
    for k, v in verd.items():
        print(f"{k}  {v}", flush=True)
    for k in ("steer_transfer_transferred_best_gain", "steer_native_ratio_ratio_transferred_over_native",
              "steer_strong_to_weak_transferred_at_least_native", "steer_asymmetry_forward_ratio_exceeds_reverse",
              "steer_ceiling_ceiling_at_least_transferred_forward", "native_best_gain", "ceiling_best_gain"):
        print(f"{k}: {measured[k]}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="all", help="comma-separated 'A<-B' or all")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--min-relations", type=int, default=10)
    ap.add_argument("--max-relations", type=int, default=30)
    ap.add_argument("--n-sentences", type=int, default=200)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--smoke", action="store_true", help="32 triples, 2 alphas, 1 seed, 16 sentences")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    args.alphas = sorted(float(a) for a in str(args.alphas).split(",") if a.strip())
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if args.smoke:
        args.n, args.alphas, args.seeds, args.n_sentences = 32, args.alphas[:2], args.seeds[:1], 16
        if args.tag == "all":
            args.tag = "smoke"
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    if args.collect:
        return collect(run, args.csv_dir)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    wanted = None if args.pairs == "all" else {p.strip() for p in args.pairs.split(",")}
    for A, B, fam, kind in PAIRS:
        if wanted is not None and f"{A}<-{B}" not in wanted:
            continue
        run_pair(A, B, fam, kind, args, run, device)
    if args.smoke:
        print(f"[{NAME}] smoke run; no verdicts, no extract", flush=True)
        return 0
    return collect(run, args.csv_dir)


if __name__ == "__main__":
    sys.exit(main())
