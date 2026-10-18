# the 8 pair comparison of sec 4.2: train a fresh map straight from B's composed vectors to A's
# object vectors with labels and compare with A's transferred map, 0.969 to 0.987

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

from factgrasp import (crossmodel as X, data as D, encoders as E, gpu_guard, layers as L,
                  metrics as M, paths, registry)
from experiments.transfer.transfer_all_pairs import FIELDS, ModelSpace

sys.path.insert(0, paths.KG_CODE)
import variants as V

NAME = "fresh_map_comparison"
RUNS = os.path.join(paths.ROOT, "results", "runs")
PAIRS_PATH = os.path.join(RUNS, "transfer_all_pairs_all.json")
PAIRS = [
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "within-family"),
    ("google/gemma-4-E4B", "google/gemma-4-12B", "within-family"),
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B", "within-family"),
    ("Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B", "cross-family"),
    ("google/gemma-4-E4B", "Qwen/Qwen3.5-9B", "cross-family"),
    ("meta-llama/Llama-3.1-8B", "google/gemma-4-12B", "cross-family"),
    ("meta-llama/Llama-3.1-8B", "bert-base-uncased", "encoder-decoder"),
    ("Qwen/Qwen3.5-4B", "roberta-large", "encoder-decoder"),
]
THRESHOLDS = {
    "fresh_map": "portability = MRR(stitched) / MRR(fresh probe, 3 seeds) at least 0.9 on 7 of 8 "
           "pairs (green); 0.75 to 0.9 (amber); below 0.75 on 3 or more (red, and the paper "
           "says a retrained probe is needed, not a transferred one)",
}


def valid_parts(space: ModelSpace, splits: dict, device):
    P = (lambda t: space.proj.transform(t)) if space.proj is not None else (lambda t: t)
    tok_model = [None, None]
    emb = {}
    for f in FIELDS:
        p = E.cache_path(space.model_id, "wikidata", "valid", f, space.pooling, space.layer)
        if not os.path.exists(p) and tok_model[0] is None:
            print(f"[{NAME}] {space.model_id}: valid cache missing for {f}, loading weights",
                  flush=True)
            tok_model[0], tok_model[1] = E.load_decoder(space.model_id, device)
        emb[f] = E.load_or_embed(splits["valid"][f].astype(str).tolist(), tok_model[0],
                                 tok_model[1], [space.layer], space.pooling,
                                 model_id=space.model_id, dataset="wikidata", split="valid",
                                 field=f, device=device, progress=False)[space.layer].float()
    if tok_model[1] is not None:
        del tok_model[1]; torch.cuda.empty_cache()
    return (P(emb["subject"]) + P(emb["pred_value"])).float(), P(emb["pred_value"]).float(), \
        P(emb["object"]).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=12)
    ap.add_argument("--pairs", type=int, default=len(PAIRS), help="first N pairs (smoke)")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(args.csv_dir, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS,
                       out_dir=RUNS)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    seeds = [int(s) for s in args.seeds.split(",")]
    stored = {}
    if os.path.exists(PAIRS_PATH):
        stored = {(v["A"], v["B"]): v for v in json.load(open(PAIRS_PATH))["rows"].values()
                  if isinstance(v, dict) and "tau" in v}
    spaces, valids = {}, {}

    def space(m):
        if m not in spaces:
            t0 = time.time()
            spaces[m] = ModelSpace(m, splits, device, args)
            valids[m] = valid_parts(spaces[m], splits, device)
            print(f"[{NAME}] {m}: layer {spaces[m].layer}, native single-map MRR "
                  f"{spaces[m].mrr_native:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return spaces[m], valids[m]

    rows = []
    for A, B, group in PAIRS[:args.pairs]:
        key = registry.config_key({"A": A, "B": B})
        if run.done(key):
            rows.append(run.get(key)); continue
        t0 = time.time()
        a, (xva, rva, ova) = space(A)
        b, (xvb, rvb, ovb) = space(B)
        S = X.fit_stitch(b.x_train_cpu, a.x_train_cpu, ridge=args.ridge).to(a.x_test.device)
        mrr_st, _ = a.stitched(b, S, splits)
        fresh = []
        for seed in seeds:
            torch.manual_seed(seed)
            Db, Da = b.x_train_cpu.shape[1], a.o_train.shape[1]
            probe = V.FlexMixtureOfW(Db, Da, 1, gate_dim=Db, gate_type="softmax").to(device)
            probe = V.train_minibatch(probe, b.x_train_cpu, b.r_train.cpu(), a.o_train.cpu(),
                                      xvb, rvb, ova, device,
                                      lambda oh, o: V.infonce_loss(oh, o, args.temp),
                                      num_epochs=args.epochs, batch_size=512, patience=12,
                                      is_mixture=False)
            probe.eval()
            v, _ = L.predict(probe, b.x_test, b.r_test, device)
            fresh.append(M.evaluate(v.cpu(), a.pool_cpu, a.gold, valid=a.vlists(splits),
                                    tie="random", seed=seed)["MRR"])
        mrr_fresh = statistics.mean(fresh)
        st = stored.get((A, B))
        row = {"A": A, "B": B, "group": group, "layer_A": a.layer, "layer_B": b.layer,
               "mrr_native_A": a.mrr_native, "mrr_native_B": b.mrr_native,
               "mrr_stitched": mrr_st, "tau": X.transfer_efficiency(mrr_st, a.mrr_native),
               "mrr_stitched_stored": (st["mrr_stitched"] if st else None),
               "mrr_fresh_by_seed": fresh, "mrr_fresh": mrr_fresh,
               "mrr_fresh_sd": (statistics.stdev(fresh) if len(fresh) > 1 else 0.0),
               "portability": (mrr_st / mrr_fresh if mrr_fresh > 0 else float("nan"))}
        run.put(key, row); rows.append(row)
        print(f"[{NAME}] {A} <- {B} ({group}): stitched {mrr_st:.4f}, fresh {mrr_fresh:.4f} "
              f"(sd {row['mrr_fresh_sd']:.4f}), portability {row['portability']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if gpu_guard.check_yield():
            gpu_guard.yield_and_exit("rerun to continue")
    ports = [r["portability"] for r in rows]
    n_hi = sum(1 for p in ports if p >= 0.9)
    n_lo = sum(1 for p in ports if p < 0.75)
    verdict = ("red" if n_lo >= 3 else "green" if n_hi >= 7 else "amber")
    run.set_summary({"n_pairs": len(rows), "verdict": verdict,
                     "portability": {f"{r['A']} <- {r['B']}": round(r["portability"], 4) for r in rows},
                     "n_at_least_0.9": n_hi, "n_below_0.75": n_lo,
                     "median_portability": round(statistics.median(ports), 4) if ports else None})
    run.flush()
    if args.tag != "all":
        print(f"[{NAME}] tag {args.tag}: extract not written", flush=True)
        args.csv_dir = None
    with open(os.path.join(args.csv_dir or os.devnull, "fresh_map_comparison.csv") if args.csv_dir else os.devnull, "w") as fh:
        fh.write("A,B,group,layer_A,layer_B,mrr_native_A,mrr_native_B,mrr_stitched,tau,"
                 "mrr_fresh,mrr_fresh_sd,portability\n")
        for r in rows:
            fh.write(f"{r['A']},{r['B']},{r['group']},{r['layer_A']},{r['layer_B']},"
                     f"{r['mrr_native_A']:.6f},{r['mrr_native_B']:.6f},{r['mrr_stitched']:.6f},"
                     f"{r['tau']:.6f},{r['mrr_fresh']:.6f},{r['mrr_fresh_sd']:.6f},"
                     f"{r['portability']:.6f}\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"fresh_map  {verdict}   portability at least 0.9 on {n_hi} of {len(rows)}; below 0.75 on "
          f"{n_lo}", flush=True)
    for r in rows:
        print(f"  {r['A']} <- {r['B']}: {r['portability']:.3f} (stitched {r['mrr_stitched']:.4f}, "
              f"fresh {r['mrr_fresh']:.4f})", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
