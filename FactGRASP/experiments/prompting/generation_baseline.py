# table 13 missed facts: greedy 8 token zero shot generation, a fact counts as missed when the
# normalized gold string is not inside the generation

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from factgrasp import paths

sys.path.insert(0, paths.KG_CODE)
import common as C

from factgrasp import causal, data, encoders, gpu_guard, registry
from factgrasp import theory_reference as T

NAME = "generation_baseline"


@torch.no_grad()
def generate_block(model, tok, prompts, max_new_tokens: int, device):
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True,
              max_length=64).to(device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
              return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    try:
        out = model.generate(**enc, output_logits=True, **kw)
        first = out.logits[0]
    except TypeError:
        out = model.generate(**enc, output_scores=True, **kw)
        first = out.scores[0]
    new_tokens = out.sequences[:, enc["input_ids"].shape[1]:]
    gens = tok.batch_decode(new_tokens, skip_special_tokens=True)
    return gens, first.float().cpu()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="the generation baseline generative baseline (exploratory)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="wikidata")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tag", default=None, help="default: encoders.model_slug(model)")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=256, help="triples per resumable block")
    ap.add_argument("--seeds", default="0",
                    help="unused (greedy decoding); kept for the common script interface")
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="512 rows")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n = 512
    if args.tag is None:
        args.tag = encoders.model_slug(args.model) + ("_smoke" if args.smoke else "")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    C.set_seed(0)
    run = registry.Run(NAME, args.tag, config=vars(args))

    df = data.load_dataset(args.dataset, "test")
    if args.n and args.n < len(df):
        g = torch.Generator().manual_seed(0)
        idx = sorted(torch.randperm(len(df), generator=g)[:args.n].tolist())
        df = df.iloc[idx].reset_index(drop=True)
    subs = df["subject"].astype(str).tolist()
    rels = df["pred_value"].astype(str).tolist()
    objs = df["object"].astype(str).tolist()
    prompts = causal.build_prompts(subs, rels, args.dataset)

    device = gpu_guard.acquire(min_free_gb=args.min_free_gb, allow_cpu=args.allow_cpu)
    tok, model = causal.load_model_and_tokenizer(args.model, device)
    gold_ids = causal.gold_token_ids(tok, objs)

    first_seen: dict = {}
    for i, o in enumerate(objs):
        first_seen.setdefault(o, i)
    unique_names = list(first_seen.keys())
    pool_first_ids = causal.gold_token_ids(tok, unique_names)
    name_to_uid = {nm: i for i, nm in enumerate(unique_names)}
    pool_gold = torch.tensor([name_to_uid[o] for o in objs], dtype=torch.long)

    em, cont, rank_vocab, rank_pool, ties = [], [], [], [], []
    gens_all = []
    for start in range(0, len(prompts), args.block):
        stop = min(start + args.block, len(prompts))
        key = registry.config_key({"stage": "gen", "start": start, "stop": stop,
                                   "dataset": args.dataset, "n": len(prompts),
                                   "max_new_tokens": args.max_new_tokens})
        if run.done(key):
            blk = run.get(key)
        else:
            b_em, b_ct, b_rv, b_rp, b_tie, b_gen = [], [], [], [], [], []
            for s2 in range(start, stop, args.batch):
                e2 = min(s2 + args.batch, stop)
                gens, logits = generate_block(model, tok, prompts[s2:e2],
                                              args.max_new_tokens, device)
                logp = F.log_softmax(logits, dim=-1)
                gi = gold_ids[s2:e2]
                ar = torch.arange(e2 - s2)
                b_rv += ((logits > logits[ar, gi].unsqueeze(1)).sum(1) + 1).tolist()
                S = logp[:, pool_first_ids]
                pg = pool_gold[s2:e2]
                b_rp += T.ranks_from_scores(S, pg, None, "optimistic").tolist()
                b_tie += ((S == S[ar, pg].unsqueeze(1)).sum(1) - 1).tolist()
                for k, gtxt in enumerate(gens):
                    gold = objs[s2 + k]
                    b_em.append(int(causal.normalise_answer(gtxt)
                                    == causal.normalise_answer(gold)))
                    b_ct.append(int(causal.contains_gold(gtxt, gold)))
                    b_gen.append(gtxt)
                del logits, logp, S
            blk = {"exact_match": b_em, "contains_gold": b_ct, "rank_vocab": b_rv,
                   "rank_pool": b_rp, "pool_ties": b_tie, "generations": b_gen}
            run.put(key, blk)
            if gpu_guard.check_yield():
                gpu_guard.yield_and_exit(f"{NAME}: block {start}-{stop} checkpointed")
        em += blk["exact_match"]
        cont += blk["contains_gold"]
        rank_vocab += blk["rank_vocab"]
        rank_pool += blk["rank_pool"]
        ties += blk["pool_ties"]
        gens_all += blk["generations"]
        print(f"[{NAME}] {stop}/{len(prompts)} triples")

    rv = torch.tensor(rank_vocab, dtype=torch.float)
    rp = torch.tensor(rank_pool, dtype=torch.float)
    summary = {
        "model": args.model, "dataset": args.dataset, "n": int(len(em)),
        "n_relations": int(df["pred_value"].nunique()),
        "template_coverage": causal.template_coverage(rels, args.dataset),
        "max_new_tokens": args.max_new_tokens,
        "exact_match": float(sum(em) / len(em)),
        "contains_gold": float(sum(cont) / len(cont)),
        "first_token_vocab_MRR": float((1.0 / rv).mean()),
        "first_token_vocab_Hits@1": float((rv <= 1).float().mean()),
        "first_token_vocab_Hits@10": float((rv <= 10).float().mean()),
        "first_token_vocab_median_rank": float(rv.median()),
        "pool_size": int(len(unique_names)),
        "pool_MRR": float((1.0 / rp).mean()),
        "pool_Hits@1": float((rp <= 1).float().mean()),
        "pool_Hits@10": float((rp <= 10).float().mean()),
        "pool_median_rank": float(rp.median()),
        "pool_mean_first_token_ties": float(torch.tensor(ties, dtype=torch.float).mean()),
        "examples": [{"prompt": p, "gold": o, "generation": g}
                     for p, o, g in list(zip(prompts, objs, gens_all))[:20]],
    }
    run.set_summary(summary)
    run.set_verdict("n/a")
    for line in [
        "exploratory, no verdict",
        "gold token = first token of ' ' + object; the prompts are the the steering run templates",
        "the restricted-pool rank is crude: candidates sharing a first token tie and ties "
        "are broken optimistically; the mean number of tied candidates is reported",
        "greedy decoding, no few-shot examples, no chat template: a floor for the model",
    ]:
        run.note(line)
    torch.save({"rank_vocab": rv, "rank_pool": rp, "exact_match": torch.tensor(em),
                "contains_gold": torch.tensor(cont)}, run.aux_path("rows.pt"))

    rows = [{"metric": k, "value": (round(v, 4) if isinstance(v, float) else v)}
            for k, v in summary.items() if isinstance(v, (int, float))]
    run.summary(rows)


if __name__ == "__main__":
    main()
