# table 8 fine tuned bert row (0.381): bert-base fine tuned end to end with a simkgc style
# infonce bi encoder on the same split and the same candidate pool. a reference for what updating the
# model gives, not a control

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from factgrasp import data as D, gpu_guard, metrics as M, paths, registry

sys.path.insert(0, paths.KG_CODE)
import common as C

NAME = "finetuned_bert"
RUNS = os.path.join(paths.ROOT, "results", "runs")
THRESHOLDS = {"finetuned_bert": "reference row, no threshold"}


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


@torch.no_grad()
def encode(model, tok, texts, device, batch=256, max_len=32):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        h = model(**enc).last_hidden_state
        out.append(F.normalize(mean_pool(h, enc["attention_mask"]), dim=-1).float().cpu())
    return torch.cat(out, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch on 2,048 triples")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=12)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke:
        seeds, args.epochs = seeds[:1], 1
        if args.tag == "all":
            args.tag = "smoke"
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds=THRESHOLDS, out_dir=RUNS)
    device = gpu_guard.acquire(allow_cpu=args.allow_cpu, min_free_gb=args.min_free_gb)
    from transformers import AutoModel, AutoTokenizer
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    if args.smoke:
        splits["train"] = splits["train"].head(2048).reset_index(drop=True)
    import pandas as pd
    valid_map = D.valid_sets(pd.concat([D.load_dataset("wikidata", s) for s in ("train", "valid", "test")],
                                       ignore_index=True))
    q = {s: [f"{a} [SEP] {r}" for a, r in zip(df["subject"].astype(str), df["pred_value"].astype(str))]
         for s, df in splits.items()}
    subj = {s: df["subject"].astype(str).tolist() for s, df in splits.items()}
    obj = {s: df["object"].astype(str).tolist() for s, df in splits.items()}
    names, _, gold = C.build_unique_object_pool(splits["test"], torch.zeros(len(splits["test"]), 1))
    vlists = D.valid_index_lists(splits["test"], names, valid_map)
    vnames, _, vgold = C.build_unique_object_pool(splits["valid"], torch.zeros(len(splits["valid"]), 1))
    vvlists = D.valid_index_lists(splits["valid"], vnames, valid_map)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    per_seed, rr_seeds = [], []
    for seed in seeds:
        key = registry.config_key({"seed": seed, "epochs": args.epochs, "smoke": args.smoke})
        if run.done(key):
            per_seed.append(run.get(key)); continue
        torch.manual_seed(seed)
        enc_q = AutoModel.from_pretrained("bert-base-uncased").to(device)
        enc_t = AutoModel.from_pretrained("bert-base-uncased").to(device)
        opt = torch.optim.AdamW(list(enc_q.parameters()) + list(enc_t.parameters()), lr=args.lr, weight_decay=1e-4)
        n = len(q["train"]); best = (-1.0, None, -1)
        t0 = time.time()
        for epoch in range(args.epochs):
            enc_q.train(); enc_t.train()
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100 + epoch)).tolist()
            total = 0.0
            for i in range(0, n, args.batch):
                ix = perm[i:i + args.batch]
                eq = tok([q["train"][j] for j in ix], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
                et = tok([obj["train"][j] for j in ix], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
                es = tok([subj["train"][j] for j in ix], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
                hq = F.normalize(mean_pool(enc_q(**eq).last_hidden_state, eq["attention_mask"]), dim=-1)
                ht = F.normalize(mean_pool(enc_t(**et).last_hidden_state, et["attention_mask"]), dim=-1)
                hs = F.normalize(mean_pool(enc_t(**es).last_hidden_state, es["attention_mask"]), dim=-1)
                logits = hq @ ht.t() / args.temp
                self_neg = (hq * hs).sum(-1, keepdim=True) / args.temp
                logits = torch.cat([logits, self_neg], dim=1)
                same = torch.tensor([[obj["train"][a] == obj["train"][b] for b in ix] for a in ix], device=device)
                same.fill_diagonal_(False)
                logits = logits.masked_fill(torch.cat([same, torch.zeros(len(ix), 1, dtype=torch.bool, device=device)], 1), -1e4)
                loss = F.cross_entropy(logits, torch.arange(len(ix), device=device))
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(ix)
            enc_q.eval(); enc_t.eval()
            vq = encode(enc_q, tok, q["valid"], device); vp = encode(enc_t, tok, vnames, device)
            vres = M.evaluate_scores(vq @ vp.t(), vgold, valid=vvlists, tie="random", seed=seed)
            print(f"[{NAME}] seed {seed} epoch {epoch + 1}/{args.epochs}: train loss {total / n:.4f}, "
                  f"valid MRR {vres['MRR']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if vres["MRR"] > best[0]:
                best = (vres["MRR"], ({k: v.detach().cpu().clone() for k, v in enc_q.state_dict().items()},
                                      {k: v.detach().cpu().clone() for k, v in enc_t.state_dict().items()}), epoch + 1)
            if gpu_guard.check_yield():
                gpu_guard.yield_and_exit(f"{NAME}: seed {seed} epoch {epoch + 1}")
        enc_q.load_state_dict(best[1][0]); enc_t.load_state_dict(best[1][1])
        tq = encode(enc_q, tok, q["test"], device); tp = encode(enc_t, tok, names, device)
        res = M.evaluate_scores(tq @ tp.t(), gold, valid=vlists, tie="random", seed=seed)
        entry = {**M.jsonable(res), "seed": seed, "best_epoch": best[2], "valid_MRR": best[0],
                 "seconds": round(time.time() - t0)}
        per_seed.append(entry); rr_seeds.append(res["rr"].float())
        run.put(key, entry)
        print(f"[{NAME}] seed {seed}: test MRR {res['MRR']:.4f}, Hits@1 {res['Hits@1']:.4f}, Hits@10 "
              f"{res['Hits@10']:.4f} (best epoch {best[2]})", flush=True)
        del enc_q, enc_t, opt; torch.cuda.empty_cache()
    agg = M.aggregate_seeds(per_seed)
    frozen = {}
    p01 = os.path.join(paths.RUNS, "bert_controls_bert.json")
    if os.path.exists(p01):
        h = (json.load(open(p01)).get("summary") or {}).get("headline", {}).get("wikidata", {})
        frozen = {"MRR": h.get("MRR"), "Hits@1": h.get("Hits@1"), "Hits@10": h.get("Hits@10"),
                  "source": "results/runs/bert_controls_bert.json (mixture InfoNCE, layer 11, average pooling)"}
    run.set_summary({"verdicts": {"finetuned_bert": "reported"},
                     "measured": {"finetuned_bert_simkgc_MRR": round(agg["MRR"], 4), "finetuned_bert_simkgc_MRR_ci95": round(agg.get("MRR_ci95", float("nan")), 4),
                                  "finetuned_bert_simkgc_Hits1": round(agg.get("Hits@1", float("nan")), 4),
                                  "finetuned_bert_frozen_probe_MRR_same_encoder": frozen.get("MRR"),
                                  "best_epochs": [p["best_epoch"] for p in per_seed]},
                     "frozen_probe_reference": frozen})
    run.flush()
    if not args.smoke:
        os.makedirs(args.csv_dir, exist_ok=True)
        with open(os.path.join(args.csv_dir, "finetuned_bert.csv"), "w") as fh:
            fh.write("row,seed,MRR,Hits@1,Hits@10,best_epoch,valid_MRR\n")
            for p in per_seed:
                fh.write(f"simkgc_finetuned_bert_base,{p['seed']},{p['MRR']:.6f},{p['Hits@1']:.6f},{p['Hits@10']:.6f},{p['best_epoch']},{p['valid_MRR']:.6f}\n")
            fh.write(f"simkgc_finetuned_bert_base,mean,{agg['MRR']:.6f},{agg.get('Hits@1', float('nan')):.6f},{agg.get('Hits@10', float('nan')):.6f},,\n")
            if frozen:
                fh.write(f"frozen_probe_bert_base_bert_controls,mean,{frozen['MRR']:.6f},{frozen['Hits@1']:.6f},{frozen['Hits@10']:.6f},,\n")
    print(f"=== SUMMARY {NAME} ===", flush=True)
    print(f"finetuned_bert  reported   SimKGC fine-tuned bert-base: MRR {agg['MRR']:.4f} over {len(per_seed)} seeds "
          f"(Hits@1 {agg.get('Hits@1', float('nan')):.4f}); frozen probe on the same encoder: "
          f"{frozen.get('MRR', float('nan')):.4f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
