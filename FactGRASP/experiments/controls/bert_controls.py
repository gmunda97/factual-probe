# table 8 upper block on bert-base: trained MoT 0.355 against random targets, random encoder,
# majority object and character trigram baselines (factgrasp/controls.py), mean pooling, 5 seeds
# this is also where the filtered ranking with random tie breaking was fixed for everything else
# run: python -m experiments.controls.bert_controls

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import torch

from factgrasp import gpu_guard, paths, registry

if paths.KG_CODE not in sys.path:
    sys.path.insert(0, paths.KG_CODE)

NAME = "bert_controls"

THRESHOLDS = {
    "filter_check_conceptnet_filtered_MRR_min": 0.18,
    "filter_check_conceptnet_falsifier": 0.165,
    "filter_check_wikidata_abs_change_max": 0.01,
    "filtered_gap_filtered_gap_min": 0.08,
    "filtered_gap_falsifier_gap": 0.03,
    "selectivity_vs_control_task_min": 0.10,
    "selectivity_vs_random_encoder_min": 0.10,
    "selectivity_vs_majority_min": 0.05,
    "selectivity_red_within": 0.05,
    "selectivity_amber_band": [0.05, 0.10],
}


class Ctx:

    def __init__(self, dataset, pooling, layer, tok, model, device,
                 max_rows=None, enc_tag="", smoke=False):
        import common as C
        import variants as V
        from factgrasp import data as D

        self.dataset, self.pooling, self.layer = dataset, pooling, layer
        self.enc_tag = enc_tag
        self.tr = D.split_embeddings(dataset, "train", pooling, tok, model, device,
                                     max_rows, enc_tag, smoke)
        self.va = D.split_embeddings(dataset, "valid", pooling, tok, model, device,
                                     max_rows, enc_tag, smoke)
        self.te = D.split_embeddings(dataset, "test", pooling, tok, model, device,
                                     max_rows, enc_tag, smoke)
        self.hs_tr, self.hr_tr, self.ho_tr = self.tr.layer(layer)
        self.hs_va, self.hr_va, self.ho_va = self.va.layer(layer)
        self.hs_te, self.hr_te, self.ho_te = self.te.layer(layer)
        self.x_tr = V.compose(self.hs_tr, self.hr_tr, "add")
        self.x_va = V.compose(self.hs_va, self.hr_va, "add")
        self.x_te = V.compose(self.hs_te, self.hr_te, "add")
        self.dim = int(self.x_tr.shape[1])

        self.names_te, self.pool_te, self.gold_te = C.build_unique_object_pool(
            self.te.df, self.ho_te)
        self.names_va, self.pool_va, self.gold_va = C.build_unique_object_pool(
            self.va.df, self.ho_va)
        df_all = D.concat_splits([self.tr.df, self.va.df, self.te.df])
        self.valid_map = D.valid_sets(df_all)
        self.valid_te = D.valid_index_lists(self.te.df, self.names_te, self.valid_map)
        self.valid_va = D.valid_index_lists(self.va.df, self.names_va, self.valid_map)
        self.rho, self.m_te = D.multiplicity_rate(self.te.df, self.names_te, self.valid_map)

    def stats(self) -> dict:
        from factgrasp import theory_reference as T
        return {"n_train": int(len(self.tr.df)), "n_valid": int(len(self.va.df)),
                "n_test": int(len(self.te.df)), "pool_test": int(len(self.names_te)),
                "n_relations": int(self.tr.df["pred_value"].nunique()),
                "multiplicity_rate": self.rho,
                "ambiguity_ceiling": T.ambiguity_ceiling(self.m_te),
                "mean_valid_in_pool": float(self.m_te.float().mean().item())}


def train_probe(ctx, E, loss_name, seed, device, epochs, o_tr=None, o_va=None):
    import common as C
    import variants as V
    C.set_seed(seed)
    model = V.FlexMixtureOfW(ctx.dim, ctx.dim, E, gate_dim=ctx.dim).to(device)
    loss_fn = V.make_loss(loss_name, lam=1.0, temp=0.05)
    o_tr = ctx.ho_tr if o_tr is None else o_tr
    o_va = ctx.ho_va if o_va is None else o_va
    return V.train_minibatch(model, ctx.x_tr, ctx.hr_tr, o_tr, ctx.x_va, ctx.hr_va, o_va,
                             device, loss_fn=loss_fn, num_epochs=epochs,
                             is_mixture=(E > 1))


def predict(model, x, hr, device):
    with torch.no_grad():
        out = model(x.to(device), hr.to(device))
    if isinstance(out, tuple):
        return out[0].float().cpu(), out[1].float().cpu()
    return out.float().cpu(), None


def eval_probe(ctx, model, device, seed):
    from factgrasp import metrics, probes, theory_reference as T
    o_te, g_te = predict(model, ctx.x_te, ctx.hr_te, device)
    met = metrics.evaluate(o_te, ctx.pool_te, ctx.gold_te, valid=ctx.valid_te,
                           tie="random", seed=seed)
    o_va, _ = predict(model, ctx.x_va, ctx.hr_va, device)
    met_va = metrics.evaluate(o_va, ctx.pool_va, ctx.gold_va, valid=ctx.valid_va,
                              tie="random", seed=seed)
    met["val_MRR"] = met_va["MRR"]
    met["val_MRR_unfiltered"] = met_va["MRR_unfiltered"]
    met["n_params"] = probes.count_params(model)
    met.update(metrics.normalised_mrr(met["MRR_unfiltered"], ctx.m_te))
    met["filter_delta"] = met["MRR"] - met["MRR_unfiltered"]
    if g_te is not None:
        met.update(T.effective_experts(g_te))
    return met


def eval_scores_over_seeds(S, ctx, seeds):
    from factgrasp import metrics
    mets, rrs = [], []
    for s in seeds:
        m = metrics.evaluate_scores(S, ctx.gold_te, valid=ctx.valid_te, tie="random", seed=s)
        mets.append(m)
        rrs.append(m["rr"])
    agg = metrics.aggregate_seeds(mets)
    return agg, metrics.mean_rr(rrs)


def run_control_task(ctx, seeds, device, epochs, E, loss_name, run, rr_store, save):
    from factgrasp import controls, metrics
    rrs, mets = [], []
    for seed in seeds:
        key = registry.config_key({"control": "control_task", "dataset": ctx.dataset,
                                   "pooling": ctx.pooling, "seed": seed, "E": E,
                                   "loss": loss_name})
        if run.done(key) and key in rr_store:
            mets.append(run.get(key))
            rrs.append(rr_store[key])
            continue
        _, o_tr = controls.control_targets(ctx.tr.df, ctx.tr.df, ctx.ho_tr, seed=seed)
        _, o_va = controls.control_targets(ctx.va.df, ctx.tr.df, ctx.ho_tr, seed=seed)
        model = train_probe(ctx, E, loss_name, seed, device, epochs, o_tr=o_tr, o_va=o_va)
        met = eval_probe(ctx, model, device, seed)
        rr_store[key] = met["rr"]
        run.put(key, metrics.jsonable(met))
        save()
        mets.append(metrics.jsonable(met))
        rrs.append(met["rr"])
        print("[p0-1 control_task] %s/%s seed %d  filtered MRR=%.4f"
              % (ctx.dataset, ctx.pooling, seed, met["MRR"]))
    return metrics.aggregate_seeds(mets), metrics.mean_rr(rrs)


def run_random_encoder(dataset, pooling, layer, seed, device, epochs, E, loss_name,
                       run, rr_store, save, max_rows, smoke):
    from factgrasp import controls, metrics
    key = registry.config_key({"control": "random_encoder", "dataset": dataset,
                               "pooling": pooling, "seed": seed, "E": E, "loss": loss_name})
    if run.done(key) and key in rr_store:
        return run.get(key), rr_store[key]
    tok, model_r = controls.random_init_encoder("bert-base-uncased", seed=seed)
    model_r = model_r.to(device).eval()
    ctx_r = Ctx(dataset, pooling, layer, tok, model_r, device, max_rows,
                enc_tag="randinit", smoke=smoke)
    del model_r
    probe = train_probe(ctx_r, E, loss_name, seed, device, epochs)
    met = eval_probe(ctx_r, probe, device, seed)
    rr_store[key] = met["rr"]
    run.put(key, metrics.jsonable(met))
    save()
    print("[p0-1 random_encoder] %s/%s seed %d  filtered MRR=%.4f"
          % (dataset, pooling, seed, met["MRR"]))
    return metrics.jsonable(met), met["rr"]


def verdict_for(summary: dict) -> str:
    gaps = summary.get("selectivity", {})
    worst = None
    for ds in gaps.values():
        for name in ("control_task", "random_encoder", "majority", "lexical"):
            g = ds.get(name)
            if g is None:
                continue
            worst = g["diff"] if worst is None else min(worst, g["diff"])
    if worst is not None and worst < THRESHOLDS["selectivity_red_within"]:
        return "red"
    if summary.get("filter_check") and summary.get("filtered_gap") and summary.get("selectivity_pass"):
        return "green"
    if summary.get("filter_check"):
        return "amber"
    return "amber" if worst is not None and worst >= THRESHOLDS["selectivity_red_within"] else "red"


def main() -> int:
    ap = argparse.ArgumentParser(description="controls for the map's score on bert-base")
    ap.add_argument("--datasets", default="wikidata,conceptnet")
    ap.add_argument("--poolings", default="cls,avg")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--experts", default="1,8")
    ap.add_argument("--losses", default="cosine,infonce")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="bert")
    ap.add_argument("--control-experts", type=int, default=8)
    ap.add_argument("--control-loss", default="infonce")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="512 rows, 2 epochs, smoke cache")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    poolings = [p for p in args.poolings.split(",") if p]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    experts = [int(e) for e in args.experts.split(",") if e != ""]
    losses = [l for l in args.losses.split(",") if l]
    max_rows = 512 if args.smoke else None
    epochs = 2 if args.smoke else args.epochs
    tag = args.tag + ("_smoke" if args.smoke else "")

    run = registry.Run(NAME, tag, thresholds=THRESHOLDS, config={
        "datasets": datasets, "poolings": poolings, "seeds": seeds, "layer": args.layer,
        "experts": experts, "losses": losses, "epochs": epochs, "encoder": "bert-base-uncased",
        "compose": "add", "temp": 0.05, "lambda": 1.0, "tie": "random",
        "n_boot": args.n_boot, "smoke": bool(args.smoke),
        "control_config": {"E": args.control_experts, "loss": args.control_loss}})

    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    import common as C
    from factgrasp import controls, metrics, stats

    rr_path = run.aux_path("rr.pt")
    rr_store = torch.load(rr_path, map_location="cpu") if os.path.exists(rr_path) else {}

    def save_rr():
        tmp = rr_path + ".tmp"
        torch.save(rr_store, tmp)
        os.replace(tmp, rr_path)

    tok, bert = C.load_bert(device)

    rows = []
    ds_stats = {}
    t0 = time.time()
    for dataset in datasets:
        for pooling in poolings:
            ctx = Ctx(dataset, pooling, args.layer, tok, bert, device, max_rows,
                      enc_tag="", smoke=args.smoke)
            ds_stats.setdefault(dataset, ctx.stats())
            print("[p0-1] %s/%s: %s" % (dataset, pooling, ctx.stats()))
            for seed, E, loss_name in itertools.product(seeds, experts, losses):
                key = registry.config_key({"dataset": dataset, "pooling": pooling,
                                           "seed": seed, "E": E, "loss": loss_name})
                if run.done(key) and key in rr_store:
                    rows.append(dict(run.get(key), key=key))
                    continue
                model = train_probe(ctx, E, loss_name, seed, device, epochs)
                met = eval_probe(ctx, model, device, seed)
                rr_store[key] = met["rr"]
                rr_store[key + "|unf"] = met["rr_unfiltered"]
                rec = metrics.jsonable(met)
                rec.update({"dataset": dataset, "pooling": pooling, "seed": seed,
                            "E": E, "loss": loss_name})
                run.put(key, rec)
                save_rr()
                rows.append(dict(rec, key=key))
                print("[p0-1] %-10s %-3s seed %d E=%d %-8s  MRR=%.4f (unf %.4f) "
                      "H@1=%.4f val=%.4f"
                      % (dataset, pooling, seed, E, loss_name, met["MRR"],
                         met["MRR_unfiltered"], met["Hits@1"], met["val_MRR"]))
                del model
                if gpu_guard.check_yield():
                    run.note("yielded during the grid at %s" % key)
                    run.summary(rows)
                    gpu_guard.yield_and_exit("rerun to continue")
            del ctx

    def cell(dataset, pooling, E, loss_name):
        sel = [r for r in rows if r.get("dataset") == dataset and r.get("pooling") == pooling
               and r.get("E") == E and r.get("loss") == loss_name]
        return metrics.aggregate_seeds(sel) if sel else {}

    agg = {}
    for dataset in datasets:
        for pooling in poolings:
            for E in experts:
                for loss_name in losses:
                    a = cell(dataset, pooling, E, loss_name)
                    if a:
                        agg["%s|%s|E%d|%s" % (dataset, pooling, E, loss_name)] = a

    pooling_choice = {}
    for dataset in datasets:
        best, best_v = poolings[0], -1.0
        for pooling in poolings:
            a = agg.get("%s|%s|E1|infonce" % (dataset, pooling), {})
            v = a.get("val_MRR", -1.0)
            pooling_choice[dataset] = pooling_choice.get(dataset, best)
            if v > best_v:
                best, best_v = pooling, v
        pooling_choice[dataset] = best
        print("[p0-1] pooling choice for %s: %s (val filtered MRR %.4f, single map "
              "InfoNCE)" % (dataset, best, best_v))
    votes = {p: sum(1 for d in pooling_choice.values() if d == p) for p in poolings}
    pooling_choice["overall"] = max(votes, key=lambda p: votes[p])

    headline = {}
    for dataset in datasets:
        p = pooling_choice[dataset]
        headline[dataset] = agg.get("%s|%s|E%d|%s"
                                    % (dataset, p, args.control_experts, args.control_loss), {})
    summary = {"pooling_choice": pooling_choice, "dataset_stats": ds_stats,
               "headline_config": {"E": args.control_experts, "loss": args.control_loss},
               "headline": {d: {k: v for k, v in headline[d].items()
                                if k in ("MRR", "MRR_ci95", "MRR_unfiltered",
                                         "MRR_unfiltered_ci95", "Hits@1", "Hits@10",
                                         "filter_delta", "normalised_MRR", "ceiling",
                                         "MRR_lo", "MRR_hi", "n_params")}
                            for d in headline}}

    p10 = None
    if "conceptnet" in headline and "wikidata" in headline and headline["conceptnet"] and headline["wikidata"]:
        cn_f = headline["conceptnet"].get("MRR", float("nan"))
        wd_f = headline["wikidata"].get("MRR", float("nan"))
        wd_u = headline["wikidata"].get("MRR_unfiltered", float("nan"))
        cn_u = headline["conceptnet"].get("MRR_unfiltered", float("nan"))
        p10 = bool(cn_f >= THRESHOLDS["filter_check_conceptnet_filtered_MRR_min"]
                   and abs(wd_f - wd_u) < THRESHOLDS["filter_check_wikidata_abs_change_max"])
        gap = wd_f - cn_f
        p11 = bool(gap >= THRESHOLDS["filtered_gap_filtered_gap_min"])
        disjoint = bool(headline["conceptnet"].get("MRR_hi", 1.0)
                        < headline["wikidata"].get("MRR_lo", 0.0))
        summary.update({
            "filter_check": p10, "filtered_gap": p11,
            "filter_check_detail": {"conceptnet_filtered": cn_f, "conceptnet_unfiltered": cn_u,
                           "wikidata_filtered": wd_f, "wikidata_unfiltered": wd_u,
                           "wikidata_abs_change": abs(wd_f - wd_u),
                           "conceptnet_falsified": bool(cn_f < THRESHOLDS["filter_check_conceptnet_falsifier"])},
            "filtered_gap_detail": {"filtered_gap_wikidata_minus_conceptnet": gap,
                           "seed_intervals_disjoint": disjoint,
                           "falsified": bool(gap < THRESHOLDS["filtered_gap_falsifier_gap"])}})
    else:
        summary.update({"filter_check": None, "filtered_gap": None})
        run.note("filter_check and filtered_gap need both datasets in one run")

    selectivity = {}
    if not args.skip_controls:
        E, loss_name = args.control_experts, args.control_loss
        for dataset in datasets:
            pooling = pooling_choice[dataset]
            ctx = Ctx(dataset, pooling, args.layer, tok, bert, device, max_rows,
                      enc_tag="", smoke=args.smoke)
            probe_rr = metrics.mean_rr(
                [rr_store[registry.config_key({"dataset": dataset, "pooling": pooling,
                                               "seed": s, "E": E, "loss": loss_name})]
                 for s in seeds])
            gaps = {}

            ct_agg, ct_rr = run_control_task(ctx, seeds, device, epochs, E, loss_name,
                                             run, rr_store, save_rr)
            gaps["control_task"] = controls.selectivity(probe_rr, ct_rr, args.n_boot)

            re_met, re_rr = run_random_encoder(dataset, pooling, args.layer, seeds[0],
                                               device, epochs, E, loss_name, run,
                                               rr_store, save_rr, max_rows, args.smoke)
            gaps["random_encoder"] = controls.selectivity(probe_rr, re_rr, args.n_boot)

            S_maj = controls.majority_baseline(ctx.tr.df, ctx.te.df, ctx.names_te)
            maj_agg, maj_rr = eval_scores_over_seeds(S_maj, ctx, seeds)
            gaps["majority"] = controls.selectivity(probe_rr, maj_rr, args.n_boot)

            S_lex = controls.lexical_baseline(ctx.te.df, ctx.names_te)
            lex_agg, lex_rr = eval_scores_over_seeds(S_lex, ctx, seeds)
            gaps["lexical"] = controls.selectivity(probe_rr, lex_rr, args.n_boot)

            run.put(registry.config_key({"control": "majority", "dataset": dataset,
                                         "pooling": pooling}), maj_agg)
            run.put(registry.config_key({"control": "lexical", "dataset": dataset,
                                         "pooling": pooling}), lex_agg)
            selectivity[dataset] = gaps
            selectivity[dataset]["control_MRRs"] = {
                "probe": float(probe_rr.mean().item()),
                "control_task": ct_agg.get("MRR"), "random_encoder": re_met.get("MRR"),
                "majority": maj_agg.get("MRR"), "lexical": lex_agg.get("MRR")}
            for k in ("control_task", "random_encoder", "majority", "lexical"):
                g = gaps[k]
                print("[p0-1 selectivity] %-10s probe - %-14s = %+.4f  [%.4f, %.4f] %s"
                      % (dataset, k, g["diff"], g["ci_low"], g["ci_high"],
                         "significant" if g["significant"] else "n.s."))
            del ctx
            if gpu_guard.check_yield():
                run.note("yielded during the controls at %s" % dataset)
                gpu_guard.yield_and_exit("rerun to continue")

        passes = []
        for dataset, gaps in selectivity.items():
            passes.append(gaps["control_task"]["diff"] >= THRESHOLDS["selectivity_vs_control_task_min"]
                          and gaps["control_task"]["significant"])
            passes.append(gaps["random_encoder"]["diff"] >= THRESHOLDS["selectivity_vs_random_encoder_min"]
                          and gaps["random_encoder"]["significant"])
            passes.append(gaps["majority"]["diff"] >= THRESHOLDS["selectivity_vs_majority_min"]
                          and gaps["majority"]["significant"])
        summary["selectivity_pass"] = bool(passes) and all(passes)
    else:
        summary["selectivity_pass"] = None
        run.note("controls skipped (--skip-controls)")
    summary["selectivity"] = selectivity

    if "wikidata" in headline and headline["wikidata"]:
        deltas = [r["filter_delta"] for r in rows
                  if r.get("dataset") == "wikidata"
                  and r.get("pooling") == pooling_choice.get("wikidata")
                  and r.get("E") == args.control_experts
                  and r.get("loss") == args.control_loss]
        if len(deltas) > 1:
            summary["wikidata_filter_delta_tost"] = stats.tost(
                deltas, margin=THRESHOLDS["filter_check_wikidata_abs_change_max"])

    summary["runtime_minutes"] = round((time.time() - t0) / 60.0, 2)
    run.set_summary(summary)
    run.data["pooling_choice"] = pooling_choice
    run.set_verdict(verdict_for(summary))
    save_rr()

    table = []
    for dataset in datasets:
        for pooling in poolings:
            for E in experts:
                for loss_name in losses:
                    a = agg.get("%s|%s|E%d|%s" % (dataset, pooling, E, loss_name))
                    if not a:
                        continue
                    table.append({"dataset": dataset, "pool": pooling, "E": E,
                                  "loss": loss_name, "MRR_f": a.get("MRR"),
                                  "ci95": a.get("MRR_ci95"), "MRR_unf": a.get("MRR_unfiltered"),
                                  "H@1": a.get("Hits@1"), "H@10": a.get("Hits@10"),
                                  "normMRR": a.get("normalised_MRR"),
                                  "val": a.get("val_MRR"), "seeds": a.get("n_seeds")})
    for dataset, st in ds_stats.items():
        table.append({"dataset": dataset, "pool": "-", "E": "-", "loss": "rho/ceiling",
                      "MRR_f": st["multiplicity_rate"], "ci95": st["ambiguity_ceiling"],
                      "MRR_unf": "-", "H@1": "-", "H@10": "-", "normMRR": "-",
                      "val": "-", "seeds": st["n_test"]})
    for dataset, gaps in selectivity.items():
        for k in ("control_task", "random_encoder", "majority", "lexical"):
            g = gaps.get(k)
            if not g:
                continue
            table.append({"dataset": dataset, "pool": pooling_choice[dataset], "E": "gap",
                          "loss": k, "MRR_f": g["diff"], "ci95": g["ci_low"],
                          "MRR_unf": g["ci_high"], "H@1": g["probe_MRR"],
                          "H@10": g["control_MRR"], "normMRR": g["significant"],
                          "val": "-", "seeds": "-"})
    table.append({"dataset": "filter_check", "pool": summary.get("filter_check"), "E": "filtered_gap",
                  "loss": summary.get("filtered_gap"), "MRR_f": "selectivity",
                  "ci95": summary.get("selectivity_pass"), "MRR_unf": "pooling",
                  "H@1": pooling_choice.get("overall"), "H@10": "-", "normMRR": "-",
                  "val": "-", "seeds": "-"})
    run.summary(table)
    print("[p0-1] per-row reciprocal ranks: %s" % rr_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
