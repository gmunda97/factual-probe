# figure 8: relation level mrr E[m, r] for the 19 models on relations with at least 10 test triples,
# additive fit E = a_m + b_r + I (eq 8, crossmodel.additive_fit), interaction share 4.1 percent,
# best model per relation oracle 0.493 -> 0.514, and the learned router on 14 held out relations

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from factgrasp import crossmodel as X, data as D, encoders as E, paths, registry

NAME = "relation_expertise"
RUNS = os.path.join(paths.ROOT, "results", "runs")
ROW = "mixture8_infonce"

THRESHOLDS = {
    "interaction_share": "interaction share at most 0.15 on Wikidata (green); 0.15 to 0.30 (amber); above "
           "0.30 (red)",
    "oracle_bound": "G at most its bound on every graph and the bound below 0.05 MRR (green); bound "
           "0.05 to 0.10 (amber); above 0.10 (red)",
    "router": "router recovers at least 50 percent of the oracle gain on the held-out relations, "
           "the gain itself at least 0.02 (green); 25 to 50 percent (amber); below 25 percent "
           "or gain below 0.02 (red)",
}


def _roster() -> dict:
    from experiments.main_results.all_models import ROSTER
    return ROSTER


def per_triple_rr(model: str, graph: str):
    slug = E.model_slug(model)
    if graph == "wikidata":
        lad = os.path.join(paths.RUNS, f"ladder_{slug}.json")
        rrp = os.path.join(paths.RUNS, f"ladder_{slug}_rr.pt")
        if not (os.path.exists(lad) and os.path.exists(rrp)):
            return None
        best = (json.load(open(lad)).get("summary") or {}).get("best_layer")
        store = torch.load(rrp, map_location="cpu")
        keys = [k for k in store if k.startswith(f"{ROW}|L{best}|seed")]
    else:
        rrp = os.path.join(paths.ROOT, "results", "runs", f"conceptnet_ladder_{slug}_rr.pt")
        if not os.path.exists(rrp):
            return None
        store = torch.load(rrp, map_location="cpu")
        keys = [k for k in store if k.startswith(f"{graph}|{ROW}|seed")]
    if not keys:
        return None
    return torch.stack([store[k].float() for k in sorted(keys)]).mean(0)


def anchor_embeddings(relations: list) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(relations), 64):
            enc = tok(relations[i:i + 64], return_tensors="pt", padding=True, truncation=True,
                      max_length=32)
            h = bert(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append((h * m).sum(1) / m.sum(1).clamp_min(1.0))
    return torch.cat(out)


def train_router(Z: torch.Tensor, E_seen: torch.Tensor, w_seen: torch.Tensor, seed: int,
                 steps: int = 600, lr: float = 1e-2, l2: float = 1e-3) -> torch.nn.Module:
    torch.manual_seed(seed)
    gate = torch.nn.Linear(Z.shape[1], E_seen.shape[0])
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=l2)
    Zn = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)
    for _ in range(steps):
        p = torch.softmax(gate(Zn), dim=-1)
        expected = (p * E_seen.t()).sum(1)
        loss = -(w_seen * expected).sum() / w_seen.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="wikidata,conceptnet")
    ap.add_argument("--min-test", type=int, default=10)
    ap.add_argument("--router-seeds", default="0,1,2")
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS,
                       out_dir=RUNS)
    roster = _roster()
    exp_rows, route_rows, out = [], [], {}
    for graph in [g.strip() for g in args.graphs.split(",")]:
        tr, te = D.load_dataset(graph, "train"), D.load_dataset(graph, "test")
        rels_all = te["pred_value"].astype(str).tolist()
        counts = te["pred_value"].astype(str).value_counts()
        rels = sorted(r for r, n in counts.items() if n >= args.min_test)
        ridx = {r: i for i, r in enumerate(rels)}
        rows_by_rel = [torch.tensor([i for i, r in enumerate(rels_all) if r == rel])
                       for rel in rels]
        w = torch.tensor([len(ix) for ix in rows_by_rel], dtype=torch.float)
        w = w / w.sum()
        models, Em = [], []
        for m in roster:
            rr = per_triple_rr(m, graph)
            if rr is None or rr.shape[0] != len(te):
                continue
            models.append(m)
            Em.append(torch.tensor([rr[ix].mean() for ix in rows_by_rel]))
        if len(models) < 3:
            print(f"[{NAME}] {graph}: only {len(models)} models with RR files, skipped")
            continue
        Emat = torch.stack(Em)
        for mi, m in enumerate(models):
            for r in rels:
                exp_rows.append({"graph": graph, "model": m, "relation": r,
                                 "n": int(counts[r]), "MRR": float(Emat[mi, ridx[r]])})
        share = X.interaction_share(Emat, w)
        G, bound, m_star = X.routing_gain(Emat, w)
        a, b, I = X.additive_fit(Emat, w)

        held = D.unseen_relation_split(tr)
        seen_ix = [ridx[r] for r in rels if r not in held]
        held_ix = [ridx[r] for r in rels if r in held]
        Z = anchor_embeddings(rels)
        w_held = w[held_ix] / w[held_ix].sum()
        best_single_held = float((w_held * Emat[m_star, held_ix]).sum())
        oracle_held = float((w_held * Emat[:, held_ix].max(0).values).sum())
        oracle_gain_held = oracle_held - best_single_held
        routed = []
        for s in [int(x) for x in args.router_seeds.split(",")]:
            gate = train_router(Z[seen_ix], Emat[:, seen_ix], w[seen_ix], seed=s)
            with torch.no_grad():
                Zn = (Z - Z[seen_ix].mean(0)) / Z[seen_ix].std(0).clamp_min(1e-6)
                choice = gate(Zn[held_ix]).argmax(-1)
            routed.append(float((w_held * Emat[choice, held_ix]).sum()))
            for j, hi in enumerate(held_ix):
                route_rows.append({"graph": graph, "seed": s, "relation": rels[hi],
                                   "n": int(counts[rels[hi]]),
                                   "chosen_model": models[int(choice[j])],
                                   "best_single_model": models[m_star],
                                   "MRR_chosen": float(Emat[choice[j], hi]),
                                   "MRR_best_single": float(Emat[m_star, hi]),
                                   "MRR_oracle": float(Emat[:, hi].max())})
        routed_mean = sum(routed) / len(routed)
        recovered = ((routed_mean - best_single_held) / oracle_gain_held
                     if oracle_gain_held > 0 else float("nan"))
        out[graph] = {
            "models": models, "n_models": len(models), "n_relations": len(rels),
            "interaction_share": share, "G": G, "bound": bound,
            "max_abs_interaction": float(I.abs().max()),
            "best_single_model": models[m_star],
            "best_single_MRR": float((w * Emat[m_star]).sum()),
            "oracle_MRR": float((w * Emat.max(0).values).sum()),
            "held_out": {"n_relations": len(held_ix), "n_test_rows": int(sum(counts[rels[i]] for i in held_ix)),
                         "best_single_MRR": best_single_held, "oracle_MRR": oracle_held,
                         "oracle_gain": oracle_gain_held, "routed_MRR_by_seed": routed,
                         "routed_MRR": routed_mean, "recovered_share": recovered},
        }
        run.put(registry.config_key({"graph": graph}), out[graph])
        run.flush()
        print(f"[{NAME}] {graph}: {len(models)} models x {len(rels)} relations; interaction "
              f"share {share:.3f}; G {G:.4f} <= bound {bound:.4f}; best single "
              f"{models[m_star]} at {out[graph]['best_single_MRR']:.4f}, oracle "
              f"{out[graph]['oracle_MRR']:.4f}; held-out: gain {oracle_gain_held:.4f}, router "
              f"{routed_mean:.4f} recovers {recovered:.2f}", flush=True)

    with open(os.path.join(args.csv_dir, "relation_expertise.csv"), "w") as fh:
        fh.write("graph,model,relation,n,MRR\n")
        for r in exp_rows:
            fh.write(f"{r['graph']},{r['model']},\"{r['relation']}\",{r['n']},{r['MRR']:.6f}\n")
    with open(os.path.join(args.csv_dir, "relation_routing.csv"), "w") as fh:
        fh.write("graph,seed,relation,n,chosen_model,best_single_model,MRR_chosen,"
                 "MRR_best_single,MRR_oracle\n")
        for r in route_rows:
            fh.write(f"{r['graph']},{r['seed']},\"{r['relation']}\",{r['n']},{r['chosen_model']},"
                     f"{r['best_single_model']},{r['MRR_chosen']:.6f},{r['MRR_best_single']:.6f},"
                     f"{r['MRR_oracle']:.6f}\n")

    wk = out.get("wikidata", {})
    share_w = wk.get("interaction_share", float("nan"))
    v_a = ("green" if share_w <= 0.15 else "amber" if share_w <= 0.30 else "red")
    identity = all(g["G"] <= g["bound"] + 1e-9 for g in out.values())
    bmax = max(g["bound"] for g in out.values()) if out else float("nan")
    v_b = ("red" if not identity or bmax > 0.10 else "green" if bmax < 0.05 else "amber")
    rec = wk.get("held_out", {}).get("recovered_share", float("nan"))
    gain = wk.get("held_out", {}).get("oracle_gain", float("nan"))
    v_c = ("red" if not (gain >= 0.02) or rec < 0.25 else "green" if rec >= 0.5 else "amber")
    verd = {"interaction_share": v_a, "oracle_bound": v_b, "router": v_c}
    run.set_summary({"graphs": list(out), "verdicts": verd,
                     "measured": {
                         "interaction_share_interaction_share_wikidata": round(share_w, 4),
                         "oracle_bound_bound_by_graph": {g: round(v["bound"], 4) for g, v in out.items()},
                         "oracle_bound_G_by_graph": {g: round(v["G"], 4) for g, v in out.items()},
                         "router_held_out_oracle_gain_wikidata": round(gain, 4),
                         "router_router_recovered_share_wikidata": round(rec, 3),
                         "conceptnet_interaction_share": round(out.get("conceptnet", {}).get(
                             "interaction_share", float("nan")), 4)}})
    run.flush()
    print(f"=== SUMMARY {NAME} ===", flush=True)
    for k, v in verd.items():
        print(f"{k}  {v}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
