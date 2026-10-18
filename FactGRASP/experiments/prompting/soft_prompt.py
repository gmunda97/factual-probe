# table 13 soft prompt: 5 trainable prompt vectors per relation placed after the subject, model frozen

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import torch

from factgrasp import (causal, controls, data as D, encoders as E, gpu_guard, layers as L,
                  metrics as M, paths, registry, stats)
from experiments.prompting.likelihood_ranking import can_say_flags, probe_rr, stratified_sample
from experiments.prompting.prompt_baselines import _expand_cache
from experiments.post_training.steering import read_ladder, wrap_pca

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

NAME = "soft_prompt"
RUNS = os.path.join(paths.ROOT, "results", "runs")
FIELDS = ("subject", "pred_value", "object")
K_SOFT = 5


def _ple_module(model):
    for m in model.modules():
        if hasattr(m, "get_per_layer_inputs") and hasattr(m, "embed_tokens_per_layer"):
            return m
    return None


def _ple_kwargs(model, ids_for_ple: torch.Tensor) -> dict:
    tm = _ple_module(model)
    if tm is None:
        return {}
    with torch.no_grad():
        return {"per_layer_inputs": tm.get_per_layer_inputs(ids_for_ple, None)}


@torch.no_grad()
def score_cached_embeds(model, tok, prompt_embeds: torch.Tensor, cand_ids: list, device, batch: int,
                        ids_for_ple: torch.Tensor = None):
    P = prompt_embeds.shape[1]
    extra = _ple_kwargs(model, ids_for_ple) if ids_for_ple is not None else {}
    out_p = model(inputs_embeds=prompt_embeds, use_cache=True, **extra)
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
        B, Lc = len(idx), len(cand_ids[idx[0]])
        ids = torch.tensor([cand_ids[i] for i in idx], dtype=torch.long, device=device)
        att = torch.ones((B, P + Lc), dtype=torch.long, device=device)
        try:
            kv = _expand_cache(copy.deepcopy(cache), B)
            logits = model(input_ids=ids, attention_mask=att, past_key_values=kv, use_cache=True).logits
            tok_lp = torch.empty(B, Lc, device=device)
            tok_lp[:, 0] = last[ids[:, 0]]
            for j in range(1, Lc):
                lp_j = torch.log_softmax(logits[:, j - 1].float(), dim=-1)
                tok_lp[:, j] = lp_j.gather(-1, ids[:, j:j + 1]).squeeze(-1)
            del logits, kv
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 8:
                raise
            batch = max(8, batch // 2)
            groups = groups[:g] + [idx[k:k + batch] for k in range(0, len(idx), batch)] + groups[g + 1:]
            continue
        summed = tok_lp.sum(1).cpu(); idx_t = torch.tensor(idx)
        out_sum[idx_t] = summed; out[idx_t] = summed / Lc
        g += 1
    del cache
    return out, out_sum


def prompt_embeds(model, tok, subject: str, soft: torch.Tensor, device, init_ids: list = None):
    emb = model.get_input_embeddings()
    ids = torch.tensor([tok(subject)["input_ids"]], device=device)
    pad = (tok(" is", add_special_tokens=False)["input_ids"][:1] or [tok.eos_token_id])[0]
    fill = list(init_ids or [])[:soft.shape[0]]
    fill += [pad] * (soft.shape[0] - len(fill))
    ids_ple = torch.cat([ids, torch.tensor([fill], dtype=torch.long, device=device)], dim=1)
    x = torch.cat([emb(ids).float(), soft.unsqueeze(0).float()], dim=1).to(next(model.parameters()).dtype)
    return x, ids_ple


def train_soft_prompt(model, tok, relation: str, rows_tr: list, rows_va: list, tr, va, device,
                      epochs: int, lr: float, batch: int, template: str) -> tuple:
    emb = model.get_input_embeddings()
    D_emb = emb.weight.shape[1]
    tail = template.split("{s}", 1)[1] if "{s}" in template else template
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"][:K_SOFT]
    pad_id = tok(" is", add_special_tokens=False)["input_ids"][:1] or [tok.eos_token_id]
    while len(tail_ids) < K_SOFT:
        tail_ids += pad_id
    with torch.no_grad():
        soft = emb(torch.tensor(tail_ids, device=device)).float().clone()
    soft = torch.nn.Parameter(soft)
    opt = torch.optim.Adam([soft], lr=lr)
    dtype = next(model.parameters()).dtype
    subj_tr = [str(tr["subject"].iloc[j]) for j in rows_tr]
    obj_tr = [str(tr["object"].iloc[j]) for j in rows_tr]
    subj_va = [str(va["subject"].iloc[j]) for j in rows_va]
    obj_va = [str(va["object"].iloc[j]) for j in rows_va]

    def gold_lp(soft_vec, subjects, objects):
        total, n = 0.0, 0
        for i in range(0, len(subjects), batch):
            s_ids = [tok(s)["input_ids"] for s in subjects[i:i + batch]]
            o_ids = [tok(" " + o, add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for o in objects[i:i + batch]]
            B = len(s_ids); Lp = max(len(s) for s in s_ids) + K_SOFT; Lo = max(len(o) for o in o_ids)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.full((B, Lp + Lo), pad, dtype=torch.long, device=device)
            att = torch.zeros((B, Lp + Lo), dtype=torch.long, device=device)
            soft_pos = []
            for b in range(B):
                n_s = len(s_ids[b]); ids[b, :n_s] = torch.tensor(s_ids[b]); att[b, :n_s + K_SOFT] = 1
                ids[b, n_s + K_SOFT:n_s + K_SOFT + len(o_ids[b])] = torch.tensor(o_ids[b]); att[b, n_s + K_SOFT:n_s + K_SOFT + len(o_ids[b])] = 1
                soft_pos.append(n_s)
            x = emb(ids).float()
            ids_ple = ids.clone()
            for b in range(B):
                x[b, soft_pos[b]:soft_pos[b] + K_SOFT] = soft_vec
                ids_ple[b, soft_pos[b]:soft_pos[b] + K_SOFT] = torch.tensor(tail_ids, device=device)
            logits = model(inputs_embeds=x.to(dtype), attention_mask=att, **_ple_kwargs(model, ids_ple)).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for b in range(B):
                st = soft_pos[b] + K_SOFT - 1; n_o = len(o_ids[b])
                total = total + tok_lp[b, st:st + n_o].sum() / n_o
                n += 1
        return total / max(n, 1)

    best = (-1e9, soft.detach().clone(), 0)
    with torch.no_grad():
        v0 = float(gold_lp(soft, subj_va, obj_va)) if subj_va else float("nan")
    hist = [v0]
    if subj_va and v0 == v0:
        best = (v0, soft.detach().clone(), 0)
    order = list(range(len(subj_tr)))
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            ix = order[i:i + batch]
            loss = -gold_lp(soft, [subj_tr[j] for j in ix], [obj_tr[j] for j in ix])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(gold_lp(soft, subj_va, obj_va)) if subj_va else float("nan")
        hist.append(v)
        if subj_va and v == v and v > best[0]:
            best = (v, soft.detach().clone(), ep + 1)
    if not subj_va:
        best = (float("nan"), soft.detach().clone(), epochs)
    return best[1], {"relation": relation, "n_train": len(subj_tr), "n_valid": len(subj_va),
                     "valid_gold_logp_by_epoch": [round(h, 4) if h == h else None for h in hist],
                     "chosen_epoch": best[2], "init_token_ids": tail_ids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--valid-per-relation", type=int, default=20)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true", help="8 triples, 2 relations, 1 epoch on 64 rows")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.epochs, args.valid_per_relation = 8, 1, 3
    tag = args.tag or (E.model_slug(args.model) + ("_smoke" if args.smoke else ""))
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, tag, config=vars(args), out_dir=RUNS)
    df = D.load_dataset("wikidata", "test"); tr = D.load_dataset("wikidata", "train"); va = D.load_dataset("wikidata", "valid")
    rels = df["pred_value"].astype(str).tolist()
    idx07, flags = can_say_flags(args.model, len(df))
    sample = stratified_sample(rels, idx07, args.n, seed=0)
    pos07 = {i: j for j, i in enumerate(idx07)}
    can = torch.tensor([bool(flags[pos07[i]]) for i in sample])
    rr_probe_all, best_layer = probe_rr(args.model, len(df))
    rr_probe = rr_probe_all[torch.tensor(sample)]
    import pandas as pd
    valid_map = D.valid_sets(pd.concat([tr, va, df], ignore_index=True))
    names, _, gold_all = C.build_unique_object_pool(df, torch.zeros(len(df), 1))
    vlists_all = D.valid_index_lists(df, names, valid_map)
    gold = gold_all[torch.tensor(sample)]; vlists = [vlists_all[i] for i in sample]
    sel = torch.tensor(sample)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    tok, model = E.load_decoder(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    cand_ids = [tok(" " + str(n), add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for n in names]
    sample_rels = sorted({rels[i] for i in sample})
    print(f"[{NAME}] {args.model}: {len(sample)} triples, {len(sample_rels)} relations, pool {len(names)}", flush=True)

    if args.smoke:
        ids = torch.tensor([tok("The capital of France is")["input_ids"]], device=device)
        with torch.no_grad():
            a = model(input_ids=ids).logits[0, -1].float()
            b = model(inputs_embeds=model.get_input_embeddings()(ids), **_ple_kwargs(model, ids)).logits[0, -1].float()
        gap = float((a - b).abs().max())
        print(f"[{NAME}] smoke: logits from ids against input embeddings, max |diff| {gap:.2e}", flush=True)
        run.note(f"inputs_embeds path against input_ids on one prompt: max abs logit diff {gap:.2e}")
        if gap > 1e-2:
            raise SystemExit(f"[{NAME}] the embedding path does not reproduce the id path ({gap:.3e})")

    tr_by, va_by = {}, {}
    for j, r in enumerate(tr["pred_value"].astype(str).tolist()):
        tr_by.setdefault(r, []).append(j)
    for j, r in enumerate(va["pred_value"].astype(str).tolist()):
        va_by.setdefault(r, []).append(j)
    soft, detail = {}, {}
    key_soft = registry.config_key({"stage": "soft", "n": args.n, "epochs": args.epochs, "smoke": args.smoke})
    soft_path = run.aux_path("soft_vectors.pt")
    if run.done(key_soft) and os.path.exists(soft_path):
        soft = torch.load(soft_path, map_location="cpu"); detail = run.get(key_soft)["detail"]
    else:
        t0 = time.time()
        rng = random.Random(0)
        for r in (sample_rels[:2] if args.smoke else sample_rels):
            rows_tr = list(tr_by.get(r, []))
            if args.smoke:
                rows_tr = rows_tr[:64]
            rows_va = list(va_by.get(r, [])); rng.shuffle(rows_va); rows_va = rows_va[:args.valid_per_relation]
            if not rows_tr:
                detail[r] = {"relation": r, "n_train": 0, "fallback": "original template"}; continue
            tpl = causal.WIKIDATA_TEMPLATES.get(r, causal.GENERIC_TEMPLATE.replace("{r}", r))
            vec, d = train_soft_prompt(model, tok, r, rows_tr, rows_va, tr, va, device, args.epochs, args.lr,
                                       args.train_batch, tpl)
            soft[r] = vec.cpu(); detail[r] = d
            if gpu_guard.check_yield():
                torch.save(soft, soft_path)
                gpu_guard.yield_and_exit(f"{NAME}: soft prompts")
        torch.save(soft, soft_path)
        run.put(key_soft, {"detail": detail, "seconds": round(time.time() - t0), "k": K_SOFT, "lr": args.lr,
                           "train_batch": args.train_batch, "epochs": args.epochs})
        print(f"[{NAME}] soft prompts trained for {len(soft)} of {len(sample_rels)} relations in "
              f"{time.time() - t0:.0f}s; chosen epochs {sorted(set(d.get('chosen_epoch', 0) for d in detail.values()))}", flush=True)

    part = run.aux_path("partial_softprompt")
    S = torch.empty(len(sample), len(names)); start = 0
    if os.path.exists(part):
        ck = torch.load(part, map_location="cpu")
        if list(ck["sample"]) == list(sample):
            start = int(ck["done"]); S[:start] = ck["S"][:start]
            print(f"[{NAME}] soft prompt scoring: resuming from {start}/{len(sample)}", flush=True)
    t0 = time.time()
    n_fallback = 0
    for r_i, i in enumerate(sample):
        if r_i < start:
            continue
        rel = rels[i]; subject = str(df["subject"].iloc[i])
        if rel in soft:
            pe, ids_ple = prompt_embeds(model, tok, subject, soft[rel].to(device), device,
                                        init_ids=(detail.get(rel) or {}).get("init_token_ids"))
            S[r_i], _ = score_cached_embeds(model, tok, pe, cand_ids, device, args.batch, ids_for_ple=ids_ple)
        else:
            n_fallback += 1
            p_ids = torch.tensor([tok(causal.fill_template(subject, rel, "wikidata"))["input_ids"]], device=device)
            pe = model.get_input_embeddings()(p_ids)
            S[r_i], _ = score_cached_embeds(model, tok, pe, cand_ids, device, args.batch, ids_for_ple=p_ids)
        if (r_i + 1) % 50 == 0 or r_i + 1 == len(sample):
            print(f"[{NAME}] softprompt {r_i + 1}/{len(sample)} triples, {time.time() - t0:.0f}s", flush=True)
            tmp = part + ".tmp"; torch.save({"S": S[:r_i + 1], "sample": sample, "done": r_i + 1}, tmp); os.replace(tmp, part)
            if gpu_guard.check_yield():
                gpu_guard.yield_and_exit(f"{NAME}: scoring")
    rr = {"softprompt": M.evaluate_scores(S, gold, valid=vlists, tie="random", seed=0)["rr"].float()}
    del model; torch.cuda.empty_cache()

    rr["majority"] = M.evaluate_scores(controls.majority_baseline(tr, df, names)[sel], gold, valid=vlists,
                                       tie="random", seed=0)["rr"].float()
    lad = read_ladder(E.model_slug(args.model), args.model)
    layer, pooling = lad["layer"], lad["pooling"]
    ck = torch.load(os.path.join(paths.CACHE_DEC, E.model_slug(args.model), f"probe_E8_infonce_L{layer}_seed0.pt"),
                    map_location="cpu", weights_only=False)
    pca = wrap_pca(ck.get("pca"))
    P = (lambda t: pca.transform(t)) if pca is not None else (lambda t: t)
    emb = {}
    for s, frame in (("train", tr), ("valid", va), ("test", df)):
        emb[s] = {f: P(torch.load(E.cache_path(args.model, "wikidata", s, f, pooling, layer), map_location="cpu").float())
                  for f in FIELDS}
    g = torch.Generator().manual_seed(0)
    pair_key = {}
    ctrl_o = emb["train"]["object"].clone()
    for j, (s_, r_) in enumerate(zip(tr["subject"].astype(str), tr["pred_value"].astype(str))):
        k = (s_, r_)
        if k not in pair_key:
            pair_key[k] = int(torch.randint(0, emb["train"]["object"].shape[0], (1,), generator=g))
        ctrl_o[j] = emb["train"]["object"][pair_key[k]]
    x_tr = emb["train"]["subject"] + emb["train"]["pred_value"]; x_va = emb["valid"]["subject"] + emb["valid"]["pred_value"]
    x_te = emb["test"]["subject"] + emb["test"]["pred_value"]
    torch.manual_seed(0)
    m = V.FlexMixtureOfW(x_tr.shape[1], ctrl_o.shape[1], 8, gate_dim=x_tr.shape[1], gate_type="softmax").to(device)
    m = V.train_minibatch(m, x_tr, emb["train"]["pred_value"], ctrl_o, x_va, emb["valid"]["pred_value"],
                          emb["valid"]["object"], device, lambda oh, o: V.infonce_loss(oh, o, 0.05),
                          num_epochs=2 if args.smoke else 60, batch_size=512, patience=12, is_mixture=True)
    o_hat, _ = L.predict(m, x_te, emb["test"]["pred_value"], device)
    names_e, pool_e, gold_e = C.build_unique_object_pool(df, emb["test"]["object"])
    assert list(names_e) == list(names)
    res_ct = M.evaluate(o_hat, pool_e, gold_e, valid=vlists_all, tie="random", seed=0)
    rr["control_task"] = res_ct["rr"].float()[sel]
    del m
    floor_rows = os.path.join(RUNS, f"untrained_floor_all_rr_{E.model_slug(args.model)}.pt")
    if os.path.exists(floor_rows):
        rr["randinit_probe"] = torch.load(floor_rows, map_location="cpu").float()[sel]
    else:
        print(f"[{NAME}] untrained floor rows for {args.model} not on disk; the random-init control is left out", flush=True)

    out = {"model": args.model, "n": len(sample), "pool": len(names), "probe_layer": best_layer,
           "soft_prompt": {"k": K_SOFT, "relations_with_prompt": len(soft), "relations_fallback": len(sample_rels) - len(soft),
                           "triples_scored_with_fallback": n_fallback, "epochs": args.epochs, "lr": args.lr,
                           "train_batch": args.train_batch, "selection": "validation gold log-probability"},
           "overall_MRR": {c: float(v.mean()) for c, v in rr.items()}, "subsets": {}}
    out["overall_MRR"]["probe"] = float(rr_probe.mean())
    for name, mask in (("all", torch.ones(len(sample), dtype=torch.bool)), ("can_say", can), ("cannot_say", ~can)):
        mk = mask
        if int(mk.sum()) == 0:
            continue
        sub = {"n": int(mk.sum()), "probe_MRR": float(rr_probe[mk].mean())}
        for c, v in rr.items():
            sub[f"{c}_MRR"] = float(v[mk].mean())
            d = (rr_probe[mk] - v[mk]).tolist()
            ci = stats.bootstrap_ci(d, n_boot=2000, seed=0)
            sub[f"adv_vs_{c}"] = float(sum(d) / len(d)); sub[f"adv_vs_{c}_lo"] = ci.get("lo"); sub[f"adv_vs_{c}_hi"] = ci.get("hi")
        out["subsets"][name] = sub
        print(f"[{NAME}] {name:10s} n={sub['n']:4d} probe {sub['probe_MRR']:.4f} softprompt {sub['softprompt_MRR']:.4f} "
              f"majority {sub['majority_MRR']:.4f} control_task {sub['control_task_MRR']:.4f}"
              + (f" randinit_probe {sub['randinit_probe_MRR']:.4f}" if "randinit_probe_MRR" in sub else ""), flush=True)
    torch.save({"rr": rr, "rr_probe": rr_probe, "sample": sample, "can": can}, run.aux_path("rows"))
    run.put(registry.config_key({"n": args.n, "smoke": args.smoke}), out)
    run.set_summary({"model": args.model, "measured": {"overall_MRR": {k: round(v, 4) for k, v in out["overall_MRR"].items()}},
                     "note": "beats_prompts to beats_soft_prompt are computed across models by prompt_baselines --collect"})
    run.flush()
    print(f"=== SUMMARY {NAME} {tag} ===", flush=True)
    print("overall MRR: " + ", ".join(f"{k} {v:.4f}" for k, v in out["overall_MRR"].items()), flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
