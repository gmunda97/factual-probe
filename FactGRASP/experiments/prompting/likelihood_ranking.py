# table 13 zero shot column: rank every candidate of the same pool by the model's length normalized
# log prob after the prompt. also draws the 1000 triple sample the other prompting scripts reuse

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from factgrasp import causal, data as D, encoders as E, gpu_guard, metrics as M, paths, registry, stats

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "likelihood_ranking"
RUNS = os.path.join(paths.ROOT, "results", "runs")
ROW = "mixture8_infonce"
THRESHOLDS = {
    "likelihood_gap": "probe MRR minus likelihood MRR at least 0.10 on 3 of 3 models with the paired "
           "bootstrap interval excluding 0 (green); 0.05 to 0.10 (amber); below 0.05 on 2 of 3 "
           "(red, and the hidden-knowledge sentence is withdrawn)",
    "cannot_say_gap": "the probe's advantage on cannot-say triples exceeds its advantage on can-say "
           "triples on 3 of 3 models (green); 2 of 3 (amber); otherwise red",
}


def _roster() -> dict:
    from experiments.main_results.all_models import ROSTER
    return ROSTER


def stratified_sample(rels: list, pool_idx: list, n: int, seed: int = 0) -> list:
    import collections, random
    by = collections.defaultdict(list)
    for i in pool_idx:
        by[rels[i]].append(i)
    rng = random.Random(seed)
    total = len(pool_idx)
    if n <= len(by) or n >= total:
        return sorted(rng.sample(pool_idx, min(n, total)))
    take = {r: max(1, round(n * len(ix) / total)) for r, ix in by.items()}
    order = sorted(by, key=lambda r: -len(by[r]))
    while sum(take.values()) > n:
        before = sum(take.values())
        for r in order:
            if take[r] > 1 and sum(take.values()) > n:
                take[r] -= 1
        if sum(take.values()) == before:
            break
    while sum(take.values()) < n:
        before = sum(take.values())
        for r in order:
            if take[r] < len(by[r]) and sum(take.values()) < n:
                take[r] += 1
        if sum(take.values()) == before:
            break
    out = []
    for r, ix in by.items():
        ix = list(ix); rng.shuffle(ix); out += ix[:take[r]]
    return sorted(out)


def can_say_flags(model: str, n_test: int):
    slug = E.model_slug(model)
    p = os.path.join(paths.RUNS, f"generation_baseline_{slug}_rows.pt")
    if not os.path.exists(p):
        return None, None
    rows = torch.load(p, map_location="cpu", weights_only=False)
    flags = rows["contains_gold"].bool()
    g = torch.Generator().manual_seed(0)
    idx = sorted(torch.randperm(n_test, generator=g)[:len(flags)].tolist())
    return idx, flags


def probe_rr(model: str, n_test: int):
    slug = E.model_slug(model)
    lad = json.load(open(os.path.join(paths.RUNS, f"ladder_{slug}.json")))
    best = lad["summary"]["best_layer"]
    store = torch.load(os.path.join(paths.RUNS, f"ladder_{slug}_rr.pt"),
                       map_location="cpu")
    keys = sorted(k for k in store if k.startswith(f"{ROW}|L{best}|seed"))
    rr = torch.stack([store[k].float() for k in keys]).mean(0)
    assert rr.shape[0] == n_test, (rr.shape, n_test)
    return rr, best


@torch.no_grad()
def score_candidates(model, tok, prompt_ids: list, cand_ids: list, device, batch: int):
    P = len(prompt_ids)
    out = torch.empty(len(cand_ids))
    out_sum = torch.empty(len(cand_ids))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    s = 0
    while s < len(cand_ids):
        chunk = cand_ids[s:s + batch]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), P + L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), P + L), dtype=torch.long)
        for j, c in enumerate(chunk):
            ids[j, :P] = torch.tensor(prompt_ids); ids[j, P:P + len(c)] = torch.tensor(c)
            att[j, :P + len(c)] = 1
        ids, att = ids.to(device), att.to(device)
        try:
            logits = model(input_ids=ids, attention_mask=att).logits
            lp = torch.log_softmax(logits[:, P - 1:P + L - 1].float(), dim=-1)
            del logits
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 8:
                raise
            batch = max(8, batch // 2)
            print(f"[{NAME}] out of memory, batch -> {batch}", flush=True)
            continue
        tgt = ids[:, P:P + L]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = att[:, P:P + L].float()
        summed = (tok_lp * mask).sum(1)
        out_sum[s:s + len(chunk)] = summed.cpu()
        out[s:s + len(chunk)] = (summed / mask.sum(1).clamp_min(1)).cpu()
        del lp, tok_lp
        s += len(chunk)
    return out, out_sum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dataset", default="wikidata")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-think", action="store_true",
                    help="close the thinking block in the chat template so the candidate is "
                         "scored as the direct answer (Qwen3.5 opens <think> by default, D-110)")
    ap.add_argument("--cand-space", default="space", choices=["space", "none"],
                    help="tokenise the candidate with a leading space (the continuation of a "
                         "bare prompt, the default and the Phase-3 runs) or without one (the "
                         "start of an assistant turn, D-114)")
    ap.add_argument("--no-chat", action="store_true",
                    help="score a post-trained model on the bare prompt, as a base model (D-114)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    if args.collect:
        return collect(args.csv_dir)
    if not args.model:
        raise SystemExit(f"[{NAME}] --model is required unless --collect")
    tag = args.tag or E.model_slug(args.model)
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    key = registry.config_key({"dataset": args.dataset, "n": args.n})
    if run.done(key):
        print(f"[{NAME}] {args.model}: already done"); return 0

    df = D.load_dataset(args.dataset, "test")
    rels = df["pred_value"].astype(str).tolist()
    idx07, flags = can_say_flags(args.model, len(df))
    if idx07 is None:
        raise SystemExit(f"[{NAME}] no the generation baseline flags for {args.model}; run generation_baseline first")
    sample = stratified_sample(rels, idx07, args.n, seed=0)
    pos07 = {i: j for j, i in enumerate(idx07)}
    can = torch.tensor([bool(flags[pos07[i]]) for i in sample])
    rr_probe_all, best = probe_rr(args.model, len(df))
    rr_probe = rr_probe_all[torch.tensor(sample)]

    import pandas as pd
    valid_map = D.valid_sets(pd.concat([D.load_dataset(args.dataset, s)
                                        for s in ("train", "valid", "test")], ignore_index=True))
    names, _, gold_all = C.build_unique_object_pool(df, torch.zeros(len(df), 1))
    vlists_all = D.valid_index_lists(df, names, valid_map)
    gold = gold_all[torch.tensor(sample)]
    vlists = [vlists_all[i] for i in sample]

    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    tok, model = E.load_decoder(args.model, device)
    post = _roster().get(args.model, ("?", 0, "base"))[2] == "post"
    use_chat = post and getattr(tok, "chat_template", None) and not args.no_chat
    lead = " " if args.cand_space == "space" else ""
    cand_ids = [tok(lead + str(n), add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
                for n in names]
    print(f"[{NAME}] {args.model}: {len(sample)} triples ({int(can.sum())} can-say), pool "
          f"{len(names)}, chat template {'on' if use_chat else 'off'}, probe layer {best}",
          flush=True)

    S = torch.empty(len(sample), len(names))
    S_sum = torch.empty(len(sample), len(names))
    start = 0
    part = run.aux_path("partial")
    if os.path.exists(part):
        ck = torch.load(part, map_location="cpu")
        if list(ck.get("sample", [])) == list(sample) and ck["S"].shape[1] == len(names):
            start = int(ck["done"]); S[:start] = ck["S"][:start]; S_sum[:start] = ck["S_sum"][:start]
            print(f"[{NAME}] resuming from {start}/{len(sample)} triples", flush=True)
    t0 = time.time()
    for r, i in enumerate(sample):
        if r < start:
            continue
        prompt = causal.fill_template(str(df["subject"].iloc[i]), rels[i], args.dataset)
        if use_chat:
            kw = {"enable_thinking": False} if args.no_think else {}
            try:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, tokenize=False, **kw)
            except TypeError:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, tokenize=False)
            p_ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            p_ids = tok(prompt)["input_ids"]
        S[r], S_sum[r] = score_candidates(model, tok, p_ids, cand_ids, device, args.batch)
        if (r + 1) % 50 == 0:
            print(f"[{NAME}] {r + 1}/{len(sample)} triples, {time.time() - t0:.0f}s", flush=True)
            tmp = part + ".tmp"
            torch.save({"S": S[:r + 1], "S_sum": S_sum[:r + 1], "sample": sample,
                        "done": r + 1}, tmp)
            os.replace(tmp, part)
            if gpu_guard.check_yield():
                gpu_guard.yield_and_exit("partial scores saved")

    res = M.evaluate_scores(S, gold, valid=vlists, tie="random", seed=0)
    res_sum = M.evaluate_scores(S_sum, gold, valid=vlists, tie="random", seed=0)
    rr_lm = res["rr"].float()
    diff = (rr_probe - rr_lm).tolist()
    out = {"model": args.model, "n": len(sample), "pool": len(names), "probe_layer": best,
           "chat_template": bool(use_chat), "no_think": bool(args.no_think),
           "cand_space": args.cand_space, "variant": tag, "subsets": {}}
    for name, mask in (("all", torch.ones(len(sample), dtype=torch.bool)),
                       ("can_say", can), ("cannot_say", ~can)):
        m = mask
        if int(m.sum()) == 0:
            continue
        d = [x for x, keep in zip(diff, m.tolist()) if keep]
        ci = stats.bootstrap_ci(d, n_boot=2000, seed=0)
        out["subsets"][name] = {
            "n": int(m.sum()),
            "probe_MRR": float(rr_probe[m].mean()), "lm_MRR": float(rr_lm[m].mean()),
            "probe_H1": float((rr_probe[m] >= 0.999).float().mean()),
            "lm_H1": float((rr_lm[m] >= 0.999).float().mean()),
            "lm_MRR_unnormalised": float(res_sum["rr"].float()[m].mean()),
            "diff": float(sum(d) / len(d)), "lo": ci.get("lo"), "hi": ci.get("hi")}
        print(f"[{NAME}] {name:10s} n={int(m.sum()):4d} probe {out['subsets'][name]['probe_MRR']:.4f} "
              f"LM {out['subsets'][name]['lm_MRR']:.4f} diff {out['subsets'][name]['diff']:+.4f} "
              f"[{ci.get('lo', float('nan')):+.4f}, {ci.get('hi', float('nan')):+.4f}]", flush=True)
    torch.save({"rr_probe": rr_probe, "rr_lm": rr_lm, "sample": sample, "can": can},
               run.aux_path("rows"))
    if os.path.exists(part):
        os.remove(part)
    run.put(key, out)
    run.set_summary({"model": args.model, "measured": {k: round(v["diff"], 4) for k, v in out["subsets"].items()},
                     "note": "likelihood_gap and cannot_say_gap are computed across models by --collect"})
    run.flush()
    print(f"=== SUMMARY {NAME} {tag} ===", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def collect(csv_dir: str) -> int:
    import glob as _glob
    def rank(v):
        chat, nothink, sp = v.get("chat_template"), v.get("no_think"), v.get("cand_space", "space")
        if "diag" in str(v.get("variant", "")):
            return 9
        if not chat:
            return 0
        if nothink and sp == "none":
            return 1
        if nothink:
            return 2
        return 3
    rows, per_model, runs = [], {}, []
    for f in sorted(_glob.glob(os.path.join(RUNS, f"{NAME}_*.json"))):
        if "collect" in f or "smoke" in f:
            continue
        d = json.load(open(f))
        for v in d["rows"].values():
            if "subsets" not in v:
                continue
            v.setdefault("variant", os.path.basename(f)[len(NAME) + 1:-5])
            runs.append(v)
    for v in runs:
        if v["model"] not in per_model or rank(v) < rank(per_model[v["model"]]):
            per_model[v["model"]] = v
    for v in runs:
        primary = per_model[v["model"]] is v
        for sub, r in v["subsets"].items():
            rows.append({"model": v["model"], "variant": v["variant"], "primary": int(primary),
                         "subset": sub, **r})
    if not rows:
        raise SystemExit(f"[{NAME}] nothing to collect")
    os.makedirs(csv_dir, exist_ok=True)
    cols = ["model", "variant", "primary", "subset", "n", "probe_MRR", "lm_MRR", "probe_H1",
            "lm_H1", "diff", "lo", "hi", "lm_MRR_unnormalised"]
    with open(os.path.join(csv_dir, "likelihood_ranking.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                              for c in cols) + "\n")
    def variant_label(v):
        if not v.get("chat_template"):
            return "bare prompt"
        if v.get("no_think") and v.get("cand_space", "space") == "none":
            return "template with thinking block closed and candidate starting the turn"
        if v.get("no_think"):
            return "template with thinking block closed and leading space"
        return "template as shipped with leading space"
    sens = [v for v in runs if v.get("chat_template") or
            any(w.get("chat_template") and w["model"] == v["model"] for w in runs)]
    with open(os.path.join(csv_dir, "template_sensitivity.csv"), "w") as fh:
        fh.write("model,variant,setting,n_triples,subset,n,lm_MRR,lm_MRR_unnormalised,lm_H1,"
                 "probe_MRR,diff,lo,hi\n")
        for v in sorted(sens, key=lambda v: (v["model"], rank(v), v["variant"])):
            for sub, r in v["subsets"].items():
                fh.write(f"{v['model']},{v['variant']},{variant_label(v)},{v['n']},{sub},{r['n']},"
                         f"{r['lm_MRR']:.6f},{r['lm_MRR_unnormalised']:.6f},{r['lm_H1']:.6f},"
                         f"{r['probe_MRR']:.6f},{r['diff']:.6f},{r['lo']:.6f},{r['hi']:.6f}\n")
    a_ok = sum(1 for v in per_model.values()
               if v["subsets"]["all"]["diff"] >= 0.10 and (v["subsets"]["all"]["lo"] or -1) > 0)
    a_low = sum(1 for v in per_model.values() if v["subsets"]["all"]["diff"] < 0.05)
    provisional = len(per_model) < 3
    v_a = ("green" if a_ok == 3 else "red" if a_low >= 2 else
           "pending" if provisional else "amber")
    b_ok = sum(1 for v in per_model.values()
               if "cannot_say" in v["subsets"] and "can_say" in v["subsets"]
               and v["subsets"]["cannot_say"]["diff"] > v["subsets"]["can_say"]["diff"])
    b_bad = len(per_model) - b_ok
    v_b = ("green" if b_ok == 3 else "red" if b_bad >= 2 else
           "pending" if provisional else "amber")
    run = registry.Run(NAME, "collect", config={"models": sorted(per_model)},
                       thresholds=THRESHOLDS, out_dir=RUNS)
    run.put("likelihood_gap", {"models_at_or_above_0.10_with_interval": a_ok, "models_below_0.05": a_low,
                    "verdict": v_a, "detail": {m: v["subsets"]["all"] for m, v in per_model.items()}})
    run.put("cannot_say_gap", {"models_where_cannot_say_gap_exceeds_can_say": b_ok, "verdict": v_b,
                    "detail": {m: {s: v["subsets"][s]["diff"] for s in v["subsets"]}
                               for m, v in per_model.items()}})
    run.set_summary({"models": sorted(per_model), "verdicts": {"likelihood_gap": v_a, "cannot_say_gap": v_b},
                     "measured": {"likelihood_gap_diff_by_model": {m: round(v["subsets"]["all"]["diff"], 4)
                                                        for m, v in per_model.items()},
                                  "cannot_say_gap_models": f"{b_ok} of {len(per_model)}"}})
    run.flush()
    print(f"=== SUMMARY {NAME} collect ===", flush=True)
    print(f"likelihood_gap  {v_a}", flush=True); print(f"cannot_say_gap  {v_b}", flush=True)
    for m, v in per_model.items():
        s = v["subsets"]
        print(f"     {m:26s} all: probe {s['all']['probe_MRR']:.4f} LM {s['all']['lm_MRR']:.4f} "
              f"diff {s['all']['diff']:+.4f} [{s['all']['lo']:+.4f}, {s['all']['hi']:+.4f}]"
              + (f"  can-say diff {s['can_say']['diff']:+.4f} cannot-say diff "
                 f"{s['cannot_say']['diff']:+.4f}" if "can_say" in s and "cannot_say" in s else ""),
              flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
