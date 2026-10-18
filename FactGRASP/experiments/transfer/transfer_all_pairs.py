# figure 3 and the 95.9 percent median of sec 4.2
# for every ordered pair A <- B of the 19 models fit a ridge S from B's composed vectors to A's on the same
# training (s, r) pairs (eq 7, ridge 1e-2, object labels never used), push S x_B through A's fixed linear
# map and rank in A's candidate space. ratio = transferred mrr / A's own mrr, 342 pairs
# ModelSpace and FIELDS are imported by most other transfer scripts

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

from factgrasp import (crossmodel as X, data as D, encoders as E, gpu_guard, layers as L,
                  metrics as M, paths, registry, stats, theory_reference as T)

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V

from experiments.post_training.steering import wrap_pca

NAME = "transfer_all_pairs"
RUNS = os.path.join(paths.ROOT, "results", "runs")
FIELDS = ("subject", "pred_value", "object")
NAMED_PAIRS = [("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"),
               ("google/gemma-4-E4B", "google/gemma-4-12B"),
               ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B")]
THRESHOLDS = {
    "named_pairs": "tau at least 0.8 on 2 of 3 named pairs with the floor below 0.3 (green); 0.6 to "
           "0.8 (amber); below 0.6 on 2 of 3 (red)",
    "cross_family": "median tau over cross-family decoder pairs at least 0.5 (green); 0.3 to 0.5 "
           "(amber); below 0.3 (red)",
    "cka": "Spearman(tau, CKA) at least 0.5 over all ordered pairs with the bootstrap "
           "interval excluding 0.2 (green); 0.3 to 0.5 (amber); below 0.3 (red)",
    "encoder_pairs": "median tau for encoder-decoder pairs below the cross-family decoder median "
           "(green); otherwise report",
}


def _roster() -> dict:
    from experiments.main_results.all_models import ROSTER
    return ROSTER


class ModelSpace:

    def __init__(self, model_id: str, splits: dict, device, args):
        self.model_id = model_id
        slug = E.model_slug(model_id)
        lad = json.load(open(os.path.join(paths.RUNS,
                                          f"ladder_{slug}.json")))["summary"]
        self.layer, self.pooling = int(lad["best_layer"]), lad.get("pooling") or "last"
        ck = os.path.join(paths.CACHE_DEC, slug, f"probe_E8_infonce_L{self.layer}_seed0.pt")
        sd = torch.load(ck, map_location="cpu")
        self.proj = wrap_pca(sd.get("pca"))
        P = (lambda t: self.proj.transform(t)) if self.proj is not None else (lambda t: t)
        tok_model = [None, None]
        emb = {}
        for s in ("train", "test"):
            emb[s] = {}
            for f in FIELDS:
                p = E.cache_path(model_id, "wikidata", s, f, self.pooling, self.layer)
                if not os.path.exists(p) and tok_model[0] is None:
                    print(f"[{NAME}] {model_id}: cache missing for {s}/{f} at layer "
                          f"{self.layer}, loading weights", flush=True)
                    tok_model[0], tok_model[1] = E.load_decoder(model_id, device)
                emb[s][f] = E.load_or_embed(splits[s][f].astype(str).tolist(), tok_model[0],
                                            tok_model[1], [self.layer], self.pooling,
                                            model_id=model_id, dataset="wikidata", split=s,
                                            field=f, device=device,
                                            progress=False)[self.layer].float()
        self.x_train_cpu = (P(emb["train"]["subject"]) + P(emb["train"]["pred_value"])).float()
        self.x_train = self.x_train_cpu.to(device)
        self.r_train = P(emb["train"]["pred_value"]).to(device)
        self.o_train = P(emb["train"]["object"]).to(device)
        self.x_test = (P(emb["test"]["subject"]) + P(emb["test"]["pred_value"])).to(device)
        self.r_test = P(emb["test"]["pred_value"]).to(device)
        names, pool, gold = C.build_unique_object_pool(splits["test"], P(emb["test"]["object"]))
        self.pool, self.gold = pool.to(device), gold
        self.gamma = T.pool_margin(self.pool, self.gold).cpu()
        ck1 = os.path.join(paths.CACHE_DEC, slug, f"probe_E1_infonce_L{self.layer}_seed0.pt")
        D_ = self.x_train.shape[1]
        self.probe = V.FlexMixtureOfW(D_, D_, 1, gate_dim=D_, gate_type="softmax").to(device)
        if os.path.exists(ck1):
            self.probe.load_state_dict(torch.load(ck1, map_location="cpu")["state_dict"])
        else:
            torch.manual_seed(0)
            v_emb = {}
            for f in FIELDS:
                p = E.cache_path(model_id, "wikidata", "valid", f, self.pooling, self.layer)
                if not os.path.exists(p) and tok_model[0] is None:
                    tok_model[0], tok_model[1] = E.load_decoder(model_id, device)
                v_emb[f] = E.load_or_embed(splits["valid"][f].astype(str).tolist(),
                                           tok_model[0], tok_model[1], [self.layer],
                                           self.pooling, model_id=model_id, dataset="wikidata",
                                           split="valid", field=f, device=device,
                                           progress=False)[self.layer].float()
            xv = P(v_emb["subject"]) + P(v_emb["pred_value"])
            self.probe = V.train_minibatch(self.probe, self.x_train.cpu(), self.r_train.cpu(),
                                           self.o_train.cpu(), xv, P(v_emb["pred_value"]),
                                           P(v_emb["object"]), device,
                                           lambda oh, o: V.infonce_loss(oh, o, args.temp),
                                           num_epochs=args.epochs, batch_size=512,
                                           patience=12, is_mixture=False)
            if args.epochs >= 60:
                torch.save({"state_dict": self.probe.state_dict(), "D": D_, "E": 1,
                            "layer": self.layer, "pooling": self.pooling, "epochs": args.epochs},
                           ck1)
        if tok_model[1] is not None:
            del tok_model[1]; torch.cuda.empty_cache()
        self.probe.eval()
        self.u, _ = L.predict(self.probe, self.x_test, self.r_test, device)
        self.u = self.u.to(device)
        self.pool_cpu = self.pool.cpu()
        self.mrr_native = M.evaluate(self.u.cpu(), self.pool_cpu, self.gold,
                                     valid=self.vlists(splits), tie="random", seed=0)["MRR"]

    def vlists(self, splits):
        if not hasattr(self, "_vl"):
            import pandas as pd
            vm = D.valid_sets(pd.concat([splits[s] for s in ("train", "valid", "test")],
                                        ignore_index=True))
            names, _, _ = C.build_unique_object_pool(splits["test"], torch.zeros(len(splits["test"]), 1))
            self._vl = D.valid_index_lists(splits["test"], names, vm)
        return self._vl

    def stitched(self, other, S, splits, seed: int = 0):
        v, _ = L.predict(self.probe, other.x_test @ S, self.r_test, self.probe_device())
        v = v.to(self.u.device)
        return M.evaluate(v.cpu(), self.pool_cpu, self.gold, valid=self.vlists(splits),
                          tie="random", seed=seed)["MRR"], v

    def probe_device(self):
        return next(self.probe.parameters()).device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma separated; default the roster")
    ap.add_argument("--collect", action="store_true", help="verdicts from the stored rows; no GPU")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--floor-seeds", default="0,1,2")
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    if args.collect:
        return collect(args.csv_dir)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS,
                       out_dir=RUNS)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    roster = _roster()
    models = ([m.strip() for m in args.models.split(",") if m.strip()] or
              [m for m in roster
               if os.path.exists(os.path.join(paths.RUNS,
                                              f"ladder_{E.model_slug(m)}.json"))])
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    spaces = {}
    for m in models:
        t0 = time.time()
        try:
            spaces[m] = ModelSpace(m, splits, device, args)
        except SystemExit as exc:
            print(f"[{NAME}] {m}: skipped ({exc})", flush=True); continue
        print(f"[{NAME}] {m}: layer {spaces[m].layer}, native single-map MRR "
              f"{spaces[m].mrr_native:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if gpu_guard.check_yield():
            gpu_guard.yield_and_exit("rerun to continue")
    names = list(spaces)
    fam = {m: roster[m][0] for m in names}
    kind = {m: ("encoder" if roster[m][0] in ("ModernBERT",) else "decoder") for m in names}
    seeds = [int(s) for s in args.floor_seeds.split(",")]
    rows = []
    for A in names:
        for B in names:
            if A == B:
                continue
            key = registry.config_key({"A": A, "B": B})
            if run.done(key):
                rows.append(run.get(key)); continue
            a, b = spaces[A], spaces[B]
            S = X.fit_stitch(b.x_train_cpu, a.x_train_cpu, ridge=args.ridge).to(a.x_test.device)
            mrr_st, v = a.stitched(b, S, splits)
            floors = []
            for s in seeds:
                perm = X.derangement(b.x_train_cpu.shape[0], seed=s)
                Sd = X.fit_stitch(b.x_train_cpu[perm], a.x_train_cpu,
                                  ridge=args.ridge).to(a.x_test.device)
                floors.append(X.transfer_efficiency(a.stitched(b, Sd, splits, seed=s)[0],
                                                    a.mrr_native))
            h_o = a.pool[a.gold.to(a.pool.device)]
            cert = float(X.stitch_certificate(a.u.cpu(), v.cpu(), h_o.cpu(), a.gamma)
                         .float().mean())
            row = {"A": A, "B": B, "family_A": fam[A], "family_B": fam[B],
                   "kind_A": kind[A], "kind_B": kind[B],
                   "tau": X.transfer_efficiency(mrr_st, a.mrr_native),
                   "floor": sum(floors) / len(floors), "floor_by_seed": floors,
                   "cert_rate": cert, "cka": X.linear_cka(a.x_test.cpu(), b.x_test.cpu()),
                   "mrr_native": a.mrr_native, "mrr_stitched": mrr_st}
            run.put(key, row); rows.append(row)
        run.flush()
        print(f"[{NAME}] {A} <- all: {len([r for r in rows if r['A'] == A])} pairs done", flush=True)
        if gpu_guard.check_yield():
            gpu_guard.yield_and_exit("rerun to continue")

    return finish(run, rows, names, args.csv_dir)


def finish(run, rows, names, csv_dir):
    roster = _roster()
    fam = {m: roster[m][0] for m in names}
    kind = {m: ("encoder" if roster[m][0] in ("ModernBERT",) else "decoder") for m in names}
    r2 = {}
    ctl = os.path.join(RUNS, "transfer_controls_all.json")
    if os.path.exists(ctl):
        for v in json.load(open(ctl))["rows"].values():
            if isinstance(v, dict) and v.get("part") == "e":
                r2[(v["A"], v["B"])] = v["r2_test"]
    with open(os.path.join(csv_dir, "transfer_all_pairs.csv"), "w") as fh:
        fh.write("A,B,family_A,family_B,kind_A,kind_B,tau,floor,cert_rate,cka,mrr_native,"
                 "mrr_stitched,r2\n")
        for r in rows:
            v = r2.get((r["A"], r["B"]))
            fh.write(f"{r['A']},{r['B']},{r['family_A']},{r['family_B']},{r['kind_A']},"
                     f"{r['kind_B']},{r['tau']:.6f},{r['floor']:.6f},{r['cert_rate']:.6f},"
                     f"{r['cka']:.6f},{r['mrr_native']:.6f},{r['mrr_stitched']:.6f},"
                     f"{'' if v is None else f'{v:.6f}'}\n")

    by = {(r["A"], r["B"]): r for r in rows}
    named = [(by.get(p), p) for p in NAMED_PAIRS]
    named_ok = sum(1 for r, _ in named if r and r["tau"] >= 0.8 and r["floor"] < 0.3)
    named_low = sum(1 for r, _ in named if r and r["tau"] < 0.6)
    v_a = ("green" if named_ok >= 2 else "red" if named_low >= 2 else "amber")
    cross = [r["tau"] for r in rows if r["kind_A"] == r["kind_B"] == "decoder"
             and r["family_A"] != r["family_B"]]
    med_cross = statistics.median(cross) if cross else float("nan")
    v_b = ("green" if med_cross >= 0.5 else "amber" if med_cross >= 0.3 else "red")
    taus = [r["tau"] for r in rows]; ckas = [r["cka"] for r in rows]
    rho = stats.spearman(taus, ckas)
    ci = stats.spearman_ci(taus, ckas, n_boot=2000)
    lo = ci.get("lo", ci.get("ci_low", float("nan"))) if isinstance(ci, dict) else float("nan")
    v_c = ("green" if rho >= 0.5 and lo > 0.2 else "amber" if rho >= 0.3 else "red")
    enc_dec = [r["tau"] for r in rows if {r["kind_A"], r["kind_B"]} == {"encoder", "decoder"}]
    med_ed = statistics.median(enc_dec) if enc_dec else float("nan")
    v_d = ("green" if enc_dec and med_ed < med_cross else "report")
    verd = {"named_pairs": v_a, "cross_family": v_b, "cka": v_c, "encoder_pairs": v_d}
    run.set_summary({"models": names, "n_pairs": len(rows), "verdicts": verd,
                     "measured": {
                         "named_pairs_named_pairs": {f"{p[0]} <- {p[1]}": (round(r["tau"], 3) if r else None)
                                             for r, p in named},
                         "named_pairs_named_floors": {f"{p[0]} <- {p[1]}": (round(r["floor"], 3) if r else None)
                                              for r, p in named},
                         "cross_family_median_cross_family_decoder_tau": round(med_cross, 4),
                         "cka_spearman_tau_cka": round(rho, 4), "cka_ci_low": lo,
                         "encoder_pairs_median_encoder_decoder_tau": (round(med_ed, 4) if enc_dec else None),
                         "median_tau_all": round(statistics.median(taus), 4),
                         "median_floor_all": round(statistics.median(r["floor"] for r in rows), 4)}})
    run.flush()
    print(f"=== SUMMARY {NAME} ===", flush=True)
    for k, v in verd.items():
        print(f"{k}  {v}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def collect(csv_dir: str) -> int:
    run = registry.Run(NAME, "all", config={}, thresholds=THRESHOLDS, out_dir=RUNS)
    rows = [v for v in run.rows_dict().values() if isinstance(v, dict) and "tau" in v] \
        if hasattr(run, "rows_dict") else \
        [v for v in json.load(open(run.out_path))["rows"].values()
         if isinstance(v, dict) and "tau" in v]
    names = sorted({r["A"] for r in rows} | {r["B"] for r in rows})
    return finish(run, rows, names, csv_dir)


if __name__ == "__main__":
    sys.exit(main())
