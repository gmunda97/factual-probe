# table 8 lower block (dataset with unique triples): resplit so no test subject or object string
# shows up in training, then the 8 map MoT with the contrastive vs the cosine loss on qwen3.5-4b smollm3-3b
# and gemma-4-e4b

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

from factgrasp import (data as D, encoders as E, gpu_guard, layers as L, metrics as M, paths,
                  registry)
from factgrasp import theory_reference as T
from factgrasp.objectives import make_objective

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

NAME = "entity_disjoint_split"
RUNS = os.path.join(paths.ROOT, "results", "runs")
AUDIT = os.path.join(RUNS, "entity_disjoint_split_audit.json")
FIELDS = ("subject", "pred_value", "object")
DECODERS = ["Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B", "google/gemma-4-E4B"]
BERT = "bert-base-uncased"
THRESHOLDS = {
    "entity_disjoint": "the level and the objective gap survive an inductive protocol: MRR at least 0.20 and "
           "InfoNCE minus cosine with interval excluding 0 on 3 of 3 decoders (green); 2 of 3 "
           "(amber); otherwise (red); read on the mixture row",
}


def _norm(s) -> str:
    return str(s).strip().lower()


def build_split(seed: int = 0, ps=(0.25, 0.30, 0.35), min_test: int = 3000) -> dict:
    import pandas as pd
    frames = []
    for s in ("train", "valid", "test"):
        df = D.load_dataset("wikidata", s).copy()
        df["src_split"] = s; df["src_row"] = range(len(df))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    ents = sorted({_norm(x) for x in all_df["subject"]} | {_norm(x) for x in all_df["object"]})
    chosen = None
    for p in ps:
        rng = random.Random(seed)
        side = {e: (rng.random() < p) for e in ents}
        test = [i for i, (s, o) in enumerate(zip(all_df["subject"], all_df["object"]))
                if side[_norm(s)] and side[_norm(o)]]
        train_side = [i for i, (s, o) in enumerate(zip(all_df["subject"], all_df["object"]))
                      if not side[_norm(s)] and not side[_norm(o)]]
        if len(test) >= min_test or p == ps[-1]:
            chosen = (p, side, test, train_side); break
    p, side, test, train_side = chosen
    rng = random.Random(seed + 1)
    tr = list(train_side); rng.shuffle(tr)
    n_val = len(tr) // 10
    valid, train = sorted(tr[:n_val]), sorted(tr[n_val:])
    dropped = len(all_df) - len(test) - len(train_side)
    return {"all_df": all_df, "p": p, "train": train, "valid": valid, "test": sorted(test),
            "n_entities": len(ents), "n_test_entities": sum(1 for v in side.values() if v),
            "dropped_mixed": dropped}


def audit_split(sp: dict) -> dict:
    df = sp["all_df"]
    tr, te = df.iloc[sp["train"]], df.iloc[sp["test"]]
    tr_ent = {_norm(x) for x in tr["subject"]} | {_norm(x) for x in tr["object"]}
    tr_trip = {(_norm(a), _norm(b), _norm(c)) for a, b, c in zip(tr["subject"], tr["pred_value"], tr["object"])}
    te_s = [_norm(x) for x in te["subject"]]; te_o = [_norm(x) for x in te["object"]]
    te_trip = [(_norm(a), _norm(b), _norm(c)) for a, b, c in zip(te["subject"], te["pred_value"], te["object"])]
    out = {"p_test_entity": sp["p"], "n_entities": sp["n_entities"], "n_test_entities": sp["n_test_entities"],
           "n_train": len(sp["train"]), "n_valid": len(sp["valid"]), "n_test": len(sp["test"]),
           "n_dropped_mixed": sp["dropped_mixed"], "n_all": len(df),
           "test_subject_seen_any_role": sum(1 for x in te_s if x in tr_ent),
           "test_object_seen_any_role": sum(1 for x in te_o if x in tr_ent),
           "test_triples_duplicated_in_train": sum(1 for t in te_trip if t in tr_trip),
           "unique_test_objects_pool": len(set(te["object"].astype(str))),
           "unique_test_relations": len(set(te["pred_value"].astype(str))),
           "test_relations_unseen_in_train": len({_norm(x) for x in te["pred_value"]} - {_norm(x) for x in tr["pred_value"]}),
           "original_test_rows_reused_in_new_test": int((te["src_split"] == "test").sum())}
    out["valid_zero_overlap"] = (out["test_subject_seen_any_role"] == 0 and out["test_object_seen_any_role"] == 0
                                 and out["test_triples_duplicated_in_train"] == 0)
    return out


def load_all_emb(model_id: str, sp: dict, device):
    df = sp["all_df"]
    if model_id == BERT:
        out = {}
        for f in FIELDS:
            parts = []
            for s in ("train", "valid", "test"):
                p = os.path.join(paths.KG_CACHE, f"{s}_{f}_avg_alllayers.pt")
                parts.append(torch.load(p, map_location="cpu")[:, 11].float())
            out[f] = torch.cat(parts, 0)
        layer, pooling = 11, "avg"
    else:
        lad = json.load(open(os.path.join(paths.RUNS, f"ladder_{E.model_slug(model_id)}.json")))["summary"]
        layer, pooling = int(lad["best_layer"]), lad.get("pooling") or "last"
        out = {}
        tok_model = [None, None]
        for f in FIELDS:
            parts = []
            for s in ("train", "valid", "test"):
                orig = D.load_dataset("wikidata", s)
                p = E.cache_path(model_id, "wikidata", s, f, pooling, layer)
                if not os.path.exists(p) and tok_model[0] is None:
                    tok_model[0], tok_model[1] = E.load_decoder(model_id, device)
                parts.append(E.load_or_embed(orig[f].astype(str).tolist(), tok_model[0], tok_model[1],
                                             [layer], pooling, model_id=model_id, dataset="wikidata",
                                             split=s, field=f, device=device, progress=False)[layer].float())
            out[f] = torch.cat(parts, 0)
        if tok_model[1] is not None:
            del tok_model[1]; torch.cuda.empty_cache()
    for f in FIELDS:
        assert out[f].shape[0] == len(df), (model_id, f, out[f].shape, len(df))
    return out, layer, pooling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DECODERS + [BERT]))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--smoke", action="store_true", help="one model, one seed, 2 epochs")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=12)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.smoke:
        models, seeds = models[:1], seeds[:1]
        if args.tag == "all":
            args.tag = "smoke"
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    sp = build_split()
    audit = audit_split(sp)
    if not args.smoke:
        json.dump(audit, open(AUDIT, "w"), indent=1)
    print(f"[{NAME}] split: p {audit['p_test_entity']}, train {audit['n_train']}, valid {audit['n_valid']}, "
          f"test {audit['n_test']} (dropped mixed {audit['n_dropped_mixed']}); overlap subjects "
          f"{audit['test_subject_seen_any_role']}, objects {audit['test_object_seen_any_role']}, duplicates "
          f"{audit['test_triples_duplicated_in_train']}; pool {audit['unique_test_objects_pool']}; "
          f"valid_zero_overlap {audit['valid_zero_overlap']}", flush=True)
    if not audit["valid_zero_overlap"]:
        raise SystemExit(f"[{NAME}] the re-split fails the zero-overlap check; see {AUDIT}")
    run.set_config({"split_audit": audit})
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    df = sp["all_df"]
    valid_map = D.valid_sets(df)
    te_df = df.iloc[sp["test"]].reset_index(drop=True)
    idx = {s: torch.tensor(sp[s], dtype=torch.long) for s in ("train", "valid", "test")}

    for model_id in models:
        t0 = time.time()
        emb, layer, pooling = load_all_emb(model_id, sp, device)
        rows_cat = torch.cat([emb[f][idx["train"]] for f in FIELDS], 0)
        proj = E.PCAProjector.fit(rows_cat, k=args.pca, seed=0) if rows_cat.shape[1] > args.pca else None
        P = (lambda t: proj.transform(t)) if proj is not None else (lambda t: t)
        Tn = {s: {f: P(emb[f][idx[s]]) for f in FIELDS} for s in idx}
        x = {s: Tn[s]["subject"] + Tn[s]["pred_value"] for s in Tn}
        names, pool, gold = C.build_unique_object_pool(te_df, Tn["test"]["object"])
        vlists = D.valid_index_lists(te_df, names, valid_map)
        print(f"[{NAME}] {model_id}: layer {layer} {pooling}, width {emb['subject'].shape[1]}, pool {len(names)} "
              f"({time.time() - t0:.0f}s to load)", flush=True)
        for obj in ("cosine", "infonce_batch"):
            for kind, n_exp in (("single", 1), ("mixture", 8)):
                key = registry.config_key({"model": model_id, "objective": obj, "probe": kind, "smoke": args.smoke})
                if run.done(key):
                    continue
                per_seed, rr_seeds = [], []
                for seed in seeds:
                    torch.manual_seed(seed)
                    m = V.FlexMixtureOfW(x["train"].shape[1], Tn["train"]["object"].shape[1], n_exp,
                                         gate_dim=Tn["train"]["pred_value"].shape[1], gate_type="softmax").to(device)
                    m = V.train_minibatch(m, x["train"], Tn["train"]["pred_value"], Tn["train"]["object"],
                                          x["valid"], Tn["valid"]["pred_value"], Tn["valid"]["object"], device,
                                          make_objective(obj, 0.05, 0.1, None),
                                          num_epochs=2 if args.smoke else args.epochs, batch_size=512,
                                          patience=8, is_mixture=(n_exp > 1))
                    o_hat, _ = L.predict(m, x["test"], Tn["test"]["pred_value"], device)
                    res = M.evaluate(o_hat, pool, gold, valid=vlists, tie="random", tau=0.05, seed=seed)
                    per_seed.append({**M.jsonable(res), "seed": seed}); rr_seeds.append(res["rr"].float())
                    print(f"[{NAME}] {model_id} {obj:14s} {kind:7s} seed {seed} MRR {res['MRR']:.4f} "
                          f"Hits@1 {res['Hits@1']:.4f}", flush=True)
                    del m
                agg = M.aggregate_seeds(per_seed)
                agg.update({"model": model_id, "objective": obj, "probe": kind, "layer": layer,
                            "pooling": pooling, "per_seed": per_seed, "pool": len(names), "n_test": len(te_df)})
                run.put(key, agg)
                torch.save(torch.stack(rr_seeds).mean(0), run.aux_path(f"rr_{E.model_slug(model_id)}_{obj}_{kind}.pt"))
                if gpu_guard.check_yield():
                    gpu_guard.yield_and_exit(f"{NAME}: {model_id}")
        del emb
    if args.smoke:
        print(f"[{NAME}] smoke run; no verdicts", flush=True)
        return 0
    return collect(run, args.csv_dir)


def collect(run, csv_dir: str) -> int:
    cells = [v for v in run.rows().values() if isinstance(v, dict) and "objective" in v]
    by = {(v["model"], v["objective"], v["probe"]): v for v in cells}
    rows, verdict_cells = [], {}
    for model_id in DECODERS + [BERT]:
        for kind in ("mixture", "single"):
            a, c = by.get((model_id, "infonce_batch", kind)), by.get((model_id, "cosine", kind))
            if not (a and c):
                continue
            pa = run.aux_path(f"rr_{E.model_slug(model_id)}_infonce_batch_{kind}.pt")
            pc = run.aux_path(f"rr_{E.model_slug(model_id)}_cosine_{kind}.pt")
            ci = (T.paired_bootstrap(torch.load(pa), torch.load(pc), n_boot=2000, seed=0)
                  if os.path.exists(pa) and os.path.exists(pc) else
                  {"diff": a["MRR"] - c["MRR"], "ci_low": float("nan"), "ci_high": float("nan")})
            row = {"model": model_id, "probe": kind, "MRR_infonce": a["MRR"], "MRR_infonce_ci95": a.get("MRR_ci95"),
                   "MRR_cosine": c["MRR"], "MRR_cosine_ci95": c.get("MRR_ci95"), "gap": ci["diff"],
                   "gap_lo": ci["ci_low"], "gap_hi": ci["ci_high"], "Hits1_infonce": a.get("Hits@1"),
                   "Hits1_cosine": c.get("Hits@1"), "pool": a["pool"], "n_test": a["n_test"], "layer": a["layer"]}
            rows.append(row)
            if kind == "mixture" and model_id in DECODERS:
                verdict_cells[model_id] = row
    ok = sum(1 for r in verdict_cells.values() if r["MRR_infonce"] >= 0.20 and r["gap_lo"] > 0)
    pending = len(verdict_cells) < 3
    v = "pending" if pending else ("green" if ok == 3 else "amber" if ok == 2 else "red")
    os.makedirs(csv_dir, exist_ok=True)
    cols = ["model", "probe", "MRR_infonce", "MRR_infonce_ci95", "MRR_cosine", "MRR_cosine_ci95", "gap", "gap_lo",
            "gap_hi", "Hits1_infonce", "Hits1_cosine", "pool", "n_test", "layer"]
    with open(os.path.join(csv_dir, "entity_disjoint_split.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")
    audit = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    run.set_summary({"verdicts": {"entity_disjoint": v},
                     "measured": {"entity_disjoint_decoders_passing": f"{ok} of {len(verdict_cells)}",
                                  "entity_disjoint_mixture_MRR_infonce": {m: round(r["MRR_infonce"], 4) for m, r in verdict_cells.items()},
                                  "entity_disjoint_mixture_gap": {m: [round(r["gap"], 4), round(r["gap_lo"], 4), round(r["gap_hi"], 4)] for m, r in verdict_cells.items()},
                                  "bert_reference": {r["probe"]: round(r["MRR_infonce"], 4) for r in rows if r["model"] == BERT},
                                  "split": {k: audit.get(k) for k in ("n_train", "n_valid", "n_test", "n_dropped_mixed",
                                                                     "unique_test_objects_pool", "p_test_entity")}}})
    run.flush()
    print(f"=== SUMMARY {NAME} collect ===", flush=True)
    print(f"entity_disjoint  {v}   " + "; ".join(f"{m}: InfoNCE {r['MRR_infonce']:.4f}, cosine {r['MRR_cosine']:.4f}, gap "
                                    f"{r['gap']:+.4f} [{r['gap_lo']:+.4f}, {r['gap_hi']:+.4f}]" for m, r in verdict_cells.items()), flush=True)
    for r in rows:
        if r["model"] == BERT:
            print(f"  bert-base-uncased {r['probe']}: InfoNCE {r['MRR_infonce']:.4f}, cosine {r['MRR_cosine']:.4f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
