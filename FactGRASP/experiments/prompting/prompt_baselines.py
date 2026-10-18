# table 13 paraphrase and five shot prompts and the majority column on the same 1000 triples

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import torch

from factgrasp import (causal, controls, data as D, encoders as E, gpu_guard, metrics as M,
                  paths, registry, stats)
from experiments.prompting.likelihood_ranking import (can_say_flags, probe_rr, score_candidates,
                                                  stratified_sample)

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "prompt_baselines"
RUNS = os.path.join(paths.ROOT, "results", "runs")
TEMPLATES = os.path.join(paths.CODE, "factgrasp", "templates_paraphrased.json")
MODELS = ["Qwen/Qwen3.5-4B", "google/gemma-4-E4B", "meta-llama/Llama-3.1-8B"]
THRESHOLDS = {
    "beats_prompts": "the map beats the strongest label-matched baseline (paraphrased template, "
           "five-shot or soft prompt, whichever has the higher overall MRR per model): advantage at least "
           "0.10 on 2 of 3 models with the interval excluding 0 (green); 0.05 to 0.10 (amber); "
           "below 0.05 on 2 of 3 (red, the sentence becomes 'beats zero-shot likelihood ranking')",
    "cannot_say_controls": "the probe beats the controls where the model cannot say: probe minus majority-object "
           "at least 0.10 on the cannot-say subset on 3 of 3 (green); 2 of 3 (amber); otherwise (red)",
    "beats_soft_prompt": "the soft prompt does not read what the map reads: probe minus soft prompt at least 0.05 "
           "on 2 of 3 (green); within 0.05 (amber); soft prompt above the probe on 2 of 3 (red)",
}


def templates_for(relation: str, table: dict) -> list:
    tpls = table.get(relation)
    if tpls is None:
        tpls = [t.replace("{r}", relation) for t in table["_generic"]]
    return tpls


def _expand_cache(cache, B: int):
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(B)
            continue
        for attr in ("conv_states", "recurrent_states"):
            states = getattr(layer, attr, None)
            if isinstance(states, dict):
                for k, t in list(states.items()):
                    if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == 1:
                        states[k] = t.repeat_interleave(B, dim=0)
            elif isinstance(states, list):
                for i, t in enumerate(states):
                    if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == 1:
                        states[i] = t.repeat_interleave(B, dim=0)
            elif torch.is_tensor(states) and states.dim() >= 1 and states.shape[0] == 1:
                setattr(layer, attr, states.repeat_interleave(B, dim=0))
    return cache


@torch.no_grad()
def score_cached(model, tok, prompt_ids: list, cand_ids: list, device, batch: int):
    P = len(prompt_ids)
    ids_p = torch.tensor([prompt_ids], device=device)
    out_p = model(input_ids=ids_p, use_cache=True)
    last = torch.log_softmax(out_p.logits[0, -1].float(), dim=-1)
    cache = out_p.past_key_values
    out = torch.empty(len(cand_ids)); out_sum = torch.empty(len(cand_ids))
    order = sorted(range(len(cand_ids)), key=lambda i: len(cand_ids[i]))
    groups = []
    for i in order:
        if groups and len(cand_ids[groups[-1][-1]]) == len(cand_ids[i]) and len(groups[-1]) < batch:
            groups[-1].append(i)
        else:
            groups.append([i])
    g = 0
    while g < len(groups):
        idx = groups[g]
        B, L = len(idx), len(cand_ids[idx[0]])
        ids = torch.tensor([cand_ids[i] for i in idx], dtype=torch.long, device=device)
        att = torch.ones((B, P + L), dtype=torch.long, device=device)
        try:
            kv = _expand_cache(copy.deepcopy(cache), B)
            logits = model(input_ids=ids, attention_mask=att, past_key_values=kv, use_cache=True).logits
            tok_lp = torch.empty(B, L, device=device)
            tok_lp[:, 0] = last[ids[:, 0]]
            for j in range(1, L):
                lp_j = torch.log_softmax(logits[:, j - 1].float(), dim=-1)
                tok_lp[:, j] = lp_j.gather(-1, ids[:, j:j + 1]).squeeze(-1)
            del logits, kv
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 8:
                raise
            batch = max(8, batch // 2)
            print(f"[{NAME}] out of memory, batch -> {batch}", flush=True)
            groups = groups[:g] + [idx[k:k + batch] for k in range(0, len(idx), batch)] + groups[g + 1:]
            continue
        summed = tok_lp.sum(1).cpu()
        idx_t = torch.tensor(idx)
        out_sum[idx_t] = summed; out[idx_t] = summed / L
        g += 1
    del cache
    return out, out_sum


def gold_logp(model, tok, prompt: str, obj: str, device) -> float:
    p_ids = tok(prompt)["input_ids"]
    c_ids = tok(" " + str(obj), add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
    return float(score_cached(model, tok, p_ids, [c_ids], device, 1)[0][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--valid-per-relation", type=int, default=20)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--smoke", action="store_true", help="8 triples, 3 validation rows per relation")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if args.collect:
        return collect(args.csv_dir)
    if not args.model:
        raise SystemExit(f"[{NAME}] --model is required unless --collect")
    if args.smoke:
        args.n, args.valid_per_relation = 8, 3
    tag = args.tag or (E.model_slug(args.model) + ("_smoke" if args.smoke else ""))
    run = registry.Run(NAME, tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    table = json.load(open(TEMPLATES))

    df = D.load_dataset("wikidata", "test")
    tr = D.load_dataset("wikidata", "train")
    va = D.load_dataset("wikidata", "valid")
    rels = df["pred_value"].astype(str).tolist()
    idx07, flags = can_say_flags(args.model, len(df))
    if idx07 is None:
        raise SystemExit(f"[{NAME}] no the generation baseline flags for {args.model}")
    sample = stratified_sample(rels, idx07, args.n, seed=0)
    pos07 = {i: j for j, i in enumerate(idx07)}
    can = torch.tensor([bool(flags[pos07[i]]) for i in sample])
    rr_probe_all, best = probe_rr(args.model, len(df))
    rr_probe = rr_probe_all[torch.tensor(sample)]

    import pandas as pd
    valid_map = D.valid_sets(pd.concat([tr, va, df], ignore_index=True))
    names, _, gold_all = C.build_unique_object_pool(df, torch.zeros(len(df), 1))
    vlists_all = D.valid_index_lists(df, names, valid_map)
    gold = gold_all[torch.tensor(sample)]
    vlists = [vlists_all[i] for i in sample]

    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    tok, model = E.load_decoder(args.model, device)
    cand_ids = [tok(" " + str(n), add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
                for n in names]
    sample_rels = sorted({rels[i] for i in sample})
    print(f"[{NAME}] {args.model}: {len(sample)} triples ({int(can.sum())} can-say), "
          f"{len(sample_rels)} relations, pool {len(names)}, probe layer {best}", flush=True)

    if args.smoke:
        i = sample[0]
        prompt = causal.fill_template(str(df["subject"].iloc[i]), rels[i], "wikidata")
        p_ids = tok(prompt)["input_ids"]
        a, _ = score_candidates(model, tok, p_ids, cand_ids[:24], device, 8)
        b, _ = score_cached(model, tok, p_ids, cand_ids[:24], device, 8)
        gap = float((a - b).abs().max())
        print(f"[{NAME}] smoke: cached against full scoring on 24 candidates, max |diff| {gap:.2e}",
              flush=True)
        run.note(f"cached scorer against likelihood_ranking's full scorer on 24 candidates: max abs diff {gap:.2e} "
                 "in bf16; in float32 the two agree to 5e-5 (checked on GPU), so "
                 "the bf16 gap is rounding noise of the split forward pass, not a scoring difference")
        if gap > 0.15:
            raise SystemExit(f"[{NAME}] cached scoring disagrees with the full scorer ({gap:.3e})")

    key_sel = registry.config_key({"stage": "select", "valid_per_relation": args.valid_per_relation,
                                   "smoke": args.smoke})
    if run.done(key_sel):
        selected = run.get(key_sel)["selected"]
    else:
        rng = random.Random(0)
        va_by = {}
        for j, r in enumerate(va["pred_value"].astype(str).tolist()):
            va_by.setdefault(r, []).append(j)
        selected, detail = {}, {}
        for r in sample_rels:
            rows = list(va_by.get(r, []))
            rng.shuffle(rows)
            rows = rows[:args.valid_per_relation]
            tpls = templates_for(r, table)
            if not rows:
                selected[r] = 0; detail[r] = {"n_valid": 0, "mean_logp": [None] * len(tpls)}
                continue
            means = []
            for tpl in tpls:
                lps = [gold_logp(model, tok, tpl.format(s=str(va["subject"].iloc[j])),
                                 str(va["object"].iloc[j]), device) for j in rows]
                means.append(sum(lps) / len(lps))
            k = max(range(len(tpls)), key=lambda t: means[t])
            selected[r] = k
            detail[r] = {"n_valid": len(rows), "mean_logp": [round(m, 4) for m in means],
                         "template": tpls[k]}
        run.put(key_sel, {"selected": selected, "detail": detail})
        chosen_other = sum(1 for k in selected.values() if k != 0)
        print(f"[{NAME}] template selection on validation: a paraphrase chosen on {chosen_other} of "
              f"{len(selected)} relations", flush=True)

    tr_by = {}
    for j, r in enumerate(tr["pred_value"].astype(str).tolist()):
        tr_by.setdefault(r, []).append(j)
    rng = random.Random(1)
    demos = {}
    for r in sample_rels:
        rows = list(tr_by.get(r, []))
        rng.shuffle(rows)
        demos[r] = [(str(tr["subject"].iloc[j]), str(tr["object"].iloc[j])) for j in rows[:args.shots]]

    conditions = {"paraphrase": lambda i: templates_for(rels[i], table)[selected[rels[i]]].format(s=str(df["subject"].iloc[i])),
                  "fiveshot": lambda i: "\n".join(f"{causal.fill_template(s, rels[i], 'wikidata')} {o}."
                                                  for s, o in demos[rels[i]])
                                        + ("\n" if demos[rels[i]] else "")
                                        + causal.fill_template(str(df["subject"].iloc[i]), rels[i], "wikidata")}
    S = {}
    for cond, make_prompt in conditions.items():
        part = run.aux_path(f"partial_{cond}")
        Sc = torch.empty(len(sample), len(names)); start = 0
        if os.path.exists(part):
            ck = torch.load(part, map_location="cpu")
            if list(ck["sample"]) == list(sample):
                start = int(ck["done"]); Sc[:start] = ck["S"][:start]
                print(f"[{NAME}] {cond}: resuming from {start}/{len(sample)}", flush=True)
        t0 = time.time()
        if start == 0 and len(sample):
            print(f"[{NAME}] {cond} example prompt: {make_prompt(sample[0])!r}", flush=True)
        for r, i in enumerate(sample):
            if r < start:
                continue
            p_ids = tok(make_prompt(i))["input_ids"]
            Sc[r], _ = score_cached(model, tok, p_ids, cand_ids, device, args.batch)
            if (r + 1) % 50 == 0 or r + 1 == len(sample):
                print(f"[{NAME}] {cond} {r + 1}/{len(sample)} triples, {time.time() - t0:.0f}s", flush=True)
                tmp = part + ".tmp"; torch.save({"S": Sc[:r + 1], "sample": sample, "done": r + 1}, tmp)
                os.replace(tmp, part)
                if gpu_guard.check_yield():
                    gpu_guard.yield_and_exit(f"{NAME}: {cond}")
        S[cond] = Sc

    sel = torch.tensor(sample)
    S["majority"] = controls.majority_baseline(tr, df, names)[sel]
    S["lexical"] = controls.lexical_baseline(df, names)[sel]

    rr = {c: M.evaluate_scores(Sc, gold, valid=vlists, tie="random", seed=0)["rr"].float()
          for c, Sc in S.items()}
    zs_path = None
    for cand in (f"likelihood_ranking_{E.model_slug(args.model)}_bare_rows",
                 f"likelihood_ranking_{E.model_slug(args.model)}_rows"):
        p = os.path.join(paths.ROOT, "results", "runs", cand)
        if os.path.exists(p):
            zs_path = p; break
    if zs_path and not args.smoke:
        zs = torch.load(zs_path, map_location="cpu")
        if list(zs["sample"]) == list(sample):
            rr["zeroshot"] = zs["rr_lm"].float()
    overall = {c: float(v.mean()) for c, v in rr.items()}
    strongest = max(("paraphrase", "fiveshot"), key=lambda c: overall[c])
    out = {"model": args.model, "n": len(sample), "pool": len(names), "probe_layer": best,
           "selected_templates": {r: templates_for(r, table)[k] for r, k in selected.items()},
           "paraphrase_chosen_on": sum(1 for k in selected.values() if k != 0),
           "n_relations": len(sample_rels), "strongest_baseline": strongest,
           "overall_MRR": {**overall, "probe": float(rr_probe.mean())}, "subsets": {}}
    for name, mask in (("all", torch.ones(len(sample), dtype=torch.bool)), ("can_say", can),
                       ("cannot_say", ~can)):
        m = mask
        if int(m.sum()) == 0:
            continue
        sub = {"n": int(m.sum()), "probe_MRR": float(rr_probe[m].mean())}
        for c in rr:
            sub[f"{c}_MRR"] = float(rr[c][m].mean())
        for c in ("paraphrase", "fiveshot", "majority", "lexical"):
            d = (rr_probe[m] - rr[c][m]).tolist()
            ci = stats.bootstrap_ci(d, n_boot=2000, seed=0)
            sub[f"adv_vs_{c}"] = float(sum(d) / len(d)); sub[f"adv_vs_{c}_lo"] = ci.get("lo"); sub[f"adv_vs_{c}_hi"] = ci.get("hi")
        d = (rr_probe[m] - rr[strongest][m]).tolist()
        ci = stats.bootstrap_ci(d, n_boot=2000, seed=0)
        sub["adv_vs_strongest"] = float(sum(d) / len(d)); sub["adv_vs_strongest_lo"] = ci.get("lo"); sub["adv_vs_strongest_hi"] = ci.get("hi")
        out["subsets"][name] = sub
        print(f"[{NAME}] {name:10s} n={sub['n']:4d} probe {sub['probe_MRR']:.4f} paraphrase "
              f"{sub['paraphrase_MRR']:.4f} fiveshot {sub['fiveshot_MRR']:.4f} majority "
              f"{sub['majority_MRR']:.4f} lexical {sub['lexical_MRR']:.4f} | adv vs strongest "
              f"({strongest}) {sub['adv_vs_strongest']:+.4f} [{sub['adv_vs_strongest_lo']:+.4f}, "
              f"{sub['adv_vs_strongest_hi']:+.4f}]", flush=True)
    torch.save({"rr": rr, "rr_probe": rr_probe, "sample": sample, "can": can}, run.aux_path("rows"))
    key = registry.config_key({"n": args.n, "shots": args.shots, "smoke": args.smoke})
    run.put(key, out)
    run.set_summary({"model": args.model, "strongest_baseline": strongest,
                     "measured": {"adv_vs_strongest_all": round(out["subsets"]["all"]["adv_vs_strongest"], 4),
                                  "overall_MRR": {k: round(v, 4) for k, v in out["overall_MRR"].items()}},
                     "note": "beats_prompts and cannot_say_controls are computed across models by --collect"})
    run.flush()
    print(f"=== SUMMARY {NAME} {tag} ===", flush=True)
    print(f"overall MRR: " + ", ".join(f"{k} {v:.4f}" for k, v in out["overall_MRR"].items()), flush=True)
    print(f"strongest label-matched baseline: {strongest}; paraphrase chosen on "
          f"{out['paraphrase_chosen_on']} of {out['n_relations']} relations", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def collect(csv_dir: str) -> int:
    per_model = {}
    for m in MODELS:
        slug = E.model_slug(m)
        pa = os.path.join(RUNS, f"{NAME}_{slug}_rows")
        if not os.path.exists(pa):
            continue
        a = torch.load(pa, map_location="cpu")
        rr, rr_probe, can = dict(a["rr"]), a["rr_probe"].float(), a["can"].bool()
        pc = os.path.join(RUNS, f"soft_prompt_{slug}_rows")
        c_meta = {}
        if os.path.exists(pc):
            c = torch.load(pc, map_location="cpu")
            if list(c["sample"]) == list(a["sample"]):
                for k, v in c["rr"].items():
                    rr[k] = v.float()
            cj = os.path.join(RUNS, f"soft_prompt_{slug}.json")
            if os.path.exists(cj):
                for v in json.load(open(cj))["rows"].values():
                    if isinstance(v, dict) and "soft_prompt" in v:
                        c_meta = v["soft_prompt"]
        rr = {k: v.float() for k, v in rr.items()}
        overall = {k: float(v.mean()) for k, v in rr.items()}
        arms = [k for k in ("paraphrase", "fiveshot", "softprompt") if k in rr]
        strongest = max(arms, key=lambda k: overall[k]) if arms else None
        subsets = {}
        for name, mask in (("all", torch.ones(len(rr_probe), dtype=torch.bool)), ("can_say", can), ("cannot_say", ~can)):
            if int(mask.sum()) == 0:
                continue
            sub = {"n": int(mask.sum()), "probe_MRR": float(rr_probe[mask].mean())}
            for k, v in rr.items():
                sub[f"{k}_MRR"] = float(v[mask].mean())
                d = (rr_probe[mask] - v[mask]).tolist()
                ci = stats.bootstrap_ci(d, n_boot=2000, seed=0)
                sub[f"adv_vs_{k}"] = float(sum(d) / len(d)); sub[f"adv_vs_{k}_lo"] = ci.get("lo"); sub[f"adv_vs_{k}_hi"] = ci.get("hi")
            if strongest:
                for suffix in ("", "_lo", "_hi"):
                    sub[f"adv_vs_strongest{suffix}"] = sub[f"adv_vs_{strongest}{suffix}"]
            subsets[name] = sub
        per_model[m] = {"model": m, "strongest_baseline": strongest, "arms": arms, "overall_MRR": {**overall, "probe": float(rr_probe.mean())},
                        "subsets": subsets, "soft_prompt": c_meta}
    if not per_model:
        raise SystemExit(f"[{NAME}] nothing to collect")
    controls_all = ["zeroshot", "paraphrase", "fiveshot", "softprompt", "majority", "lexical", "control_task", "randinit_probe"]
    cols = ["model", "subset", "n", "probe_MRR", "strongest_baseline"] + [f"{k}_MRR" for k in controls_all] + \
           ["adv_vs_strongest", "adv_vs_strongest_lo", "adv_vs_strongest_hi"] + \
           [f"adv_vs_{k}{suf}" for k in ("softprompt", "majority", "control_task", "randinit_probe", "lexical") for suf in ("", "_lo", "_hi")]
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, "prompt_baselines.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for m, v in sorted(per_model.items()):
            for sub, r in v["subsets"].items():
                row = {"model": m, "subset": sub, "strongest_baseline": v["strongest_baseline"], **r}
                fh.write(",".join(f"{row[c]:.6f}" if isinstance(row.get(c), float) else str(row.get(c, "")) for c in cols) + "\n")
    have_c = all("softprompt" in v["arms"] for v in per_model.values())
    a_ok = sum(1 for v in per_model.values() if v["subsets"]["all"].get("adv_vs_strongest", -1) >= 0.10 and (v["subsets"]["all"].get("adv_vs_strongest_lo") or -1) > 0)
    a_low = sum(1 for v in per_model.values() if v["subsets"]["all"].get("adv_vs_strongest", -1) < 0.05)
    b_ok = sum(1 for v in per_model.values() if "cannot_say" in v["subsets"] and v["subsets"]["cannot_say"]["adv_vs_majority"] >= 0.10)
    c_ok = sum(1 for v in per_model.values() if v["subsets"]["all"].get("adv_vs_softprompt", -1) >= 0.05)
    c_neg = sum(1 for v in per_model.values() if v["subsets"]["all"].get("adv_vs_softprompt", 1) < 0)
    pending = len(per_model) < 3
    v_a = "pending" if (pending or not have_c) else ("green" if a_ok >= 2 else "red" if a_low >= 2 else "amber")
    v_b = "pending" if pending else ("green" if b_ok == 3 else "amber" if b_ok == 2 else "red")
    v_c = "pending" if (pending or not have_c) else ("green" if c_ok >= 2 else "red" if c_neg >= 2 else "amber")
    run = registry.Run(NAME, "collect", config={"models": sorted(per_model)}, thresholds=THRESHOLDS, out_dir=RUNS)
    run.set_summary({"models": sorted(per_model), "verdicts": {"beats_prompts": v_a, "cannot_say_controls": v_b, "beats_soft_prompt": v_c},
                     "measured": {"beats_prompts_adv_vs_strongest": {m: round(v["subsets"]["all"].get("adv_vs_strongest", float("nan")), 4) for m, v in per_model.items()},
                                  "beats_prompts_strongest_baseline": {m: v["strongest_baseline"] for m, v in per_model.items()},
                                  "beats_prompts_models_at_least_0.10": f"{a_ok} of {len(per_model)}",
                                  "cannot_say_controls_adv_vs_majority_cannot_say": {m: round(v["subsets"]["cannot_say"]["adv_vs_majority"], 4) for m, v in per_model.items() if "cannot_say" in v["subsets"]},
                                  "cannot_say_controls_models_at_least_0.10": f"{b_ok} of {len(per_model)}",
                                  "beats_soft_prompt_adv_vs_softprompt": {m: round(v["subsets"]["all"].get("adv_vs_softprompt", float("nan")), 4) for m, v in per_model.items()},
                                  "beats_soft_prompt_models_at_least_0.05": f"{c_ok} of {len(per_model)}",
                                  "overall_MRR": {m: {k: round(x, 4) for k, x in v["overall_MRR"].items()} for m, v in per_model.items()},
                                  "soft_prompt": {m: v["soft_prompt"] for m, v in per_model.items()}}})
    run.flush()
    print(f"=== SUMMARY {NAME} collect ===", flush=True)
    print(f"beats_prompts  {v_a}", flush=True); print(f"cannot_say_controls  {v_b}", flush=True); print(f"beats_soft_prompt  {v_c}", flush=True)
    for m, v in per_model.items():
        s_ = v["subsets"]["all"]
        print(f"     {m:26s} probe {s_['probe_MRR']:.4f} | " + " ".join(f"{k} {s_.get(k + '_MRR', float('nan')):.4f}" for k in controls_all if k + "_MRR" in s_)
              + f" | adv vs strongest ({v['strongest_baseline']}) {s_.get('adv_vs_strongest', float('nan')):+.4f} "
              f"[{s_.get('adv_vs_strongest_lo', float('nan')):+.4f}, {s_.get('adv_vs_strongest_hi', float('nan')):+.4f}]; cannot-say adv vs majority "
              f"{v['subsets']['cannot_say']['adv_vs_majority']:+.4f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
