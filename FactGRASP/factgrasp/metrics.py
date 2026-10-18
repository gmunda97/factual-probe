# filtered mrr and hits@k with random tie breaking, every number in the paper goes through evaluate()

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from factgrasp import stats, theory_reference as T

TENSOR_KEYS = ("rr", "rr_unfiltered", "ranks", "ranks_unfiltered", "n_valid")


# score = cosine between the predicted vector and every candidate object vector
def cosine_scores(o_hat: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    q = F.normalize(o_hat.float(), dim=1)
    t = F.normalize(pool.float(), dim=1)
    return q @ t.t()


# filtered ranking (sec 3.8): other known correct objects of this (s, r) from train valid and test are
# taken out of the competitors, the gold stays
def _filter_scores(S: torch.Tensor, gold: torch.Tensor,
                   valid: Optional[Sequence[torch.Tensor]]) -> torch.Tensor:
    S = S.clone().float()
    if valid is None:
        return S
    for i in range(S.shape[0]):
        others = valid[i][valid[i] != gold[i]]
        if len(others):
            S[i, others] = -float("inf")
    return S


# ties broken at random so a rank is neither optimistic nor pessimistic
def random_tie_ranks(S: torch.Tensor, gold: torch.Tensor,
                     valid: Optional[Sequence[torch.Tensor]] = None,
                     seed: int = 0) -> torch.Tensor:
    S = _filter_scores(S, gold, valid)
    N = S.shape[0]
    g = S[torch.arange(N), gold].unsqueeze(1)
    greater = (S > g).sum(1)
    equal = ((S == g).sum(1) - 1).clamp_min(0)
    gen = torch.Generator().manual_seed(int(seed))
    u = torch.rand(N, generator=gen)
    offset = (u * (equal.float() + 1.0)).floor().long().clamp(max=equal)
    return greater + 1 + offset


def ranks(S: torch.Tensor, gold: torch.Tensor,
          valid: Optional[Sequence[torch.Tensor]] = None,
          tie: str = "random", seed: int = 0) -> torch.Tensor:
    if tie == "random":
        return random_tie_ranks(S, gold, valid, seed)
    return T.ranks_from_scores(S, gold, valid, tie)


def evaluate_scores(S: torch.Tensor, gold_idx: torch.Tensor,
                    valid: Optional[Sequence[torch.Tensor]] = None,
                    tie: str = "random", ks=(1, 3, 10), tau: float = 0.05,
                    seed: int = 0) -> dict:
    gold_idx = gold_idx.long()
    r_f = ranks(S, gold_idx, valid, tie, seed).float()
    r_u = ranks(S, gold_idx, None, tie, seed).float()
    out = {"MRR": (1.0 / r_f).mean().item(), "n": int(S.shape[0]),
           "pool": int(S.shape[1]), "tie": tie}
    out.update({"Hits@%d" % k: (r_f <= k).float().mean().item() for k in ks})
    out["MRR_unfiltered"] = (1.0 / r_u).mean().item()
    out.update({"Hits@%d_unfiltered" % k: (r_u <= k).float().mean().item() for k in ks})
    out["mean_rank"] = r_f.mean().item()
    sb = T.softmax_bound(S, gold_idx, tau, valid)
    out["B_tau"] = sb["B_tau"]
    out["exp_neg_L"] = sb["exp_neg_L"]
    out["gap"] = sb["gap"]
    out["bound_violations"] = sb["violations"]
    out["tau"] = tau
    out["rr"] = 1.0 / r_f
    out["rr_unfiltered"] = 1.0 / r_u
    out["ranks"] = r_f
    out["ranks_unfiltered"] = r_u
    return out


def evaluate(o_hat: torch.Tensor, pool: torch.Tensor, gold_idx: torch.Tensor,
             valid: Optional[Sequence[torch.Tensor]] = None, tie: str = "random",
             ks=(1, 3, 10), tau: float = 0.05, seed: int = 0) -> dict:
    return evaluate_scores(cosine_scores(o_hat, pool), gold_idx, valid, tie, ks, tau, seed)


def normalised_mrr(mrr_unfiltered: float, n_valid: torch.Tensor) -> dict:
    ceil = T.ambiguity_ceiling(n_valid)
    return {"ceiling": ceil, "normalised_MRR": mrr_unfiltered / ceil if ceil > 0 else float("nan")}


def jsonable(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in TENSOR_KEYS}


def aggregate_seeds(list_of_dicts: Sequence[dict], alpha: float = 0.05) -> dict:
    ds = [jsonable(d) for d in list_of_dicts]
    if not ds:
        return {}
    keys = [k for k in ds[0] if all(isinstance(d.get(k), (int, float))
                                    and not isinstance(d.get(k), bool) for d in ds)]
    out = {"n_seeds": len(ds)}
    for k in keys:
        vals = [float(d[k]) for d in ds]
        ci = stats.t_ci(vals, alpha=alpha)
        out[k] = ci["mean"]
        out[k + "_std"] = ci["sd"]
        out[k + "_ci95"] = ci["half_width"]
        out[k + "_lo"] = ci["lo"]
        out[k + "_hi"] = ci["hi"]
    return out


def paired_ci(rr_a: torch.Tensor, rr_b: torch.Tensor, n_boot: int = 2000,
              seed: int = 0, alpha: float = 0.05) -> dict:
    return T.paired_bootstrap(rr_a.float(), rr_b.float(), n_boot=n_boot, seed=seed,
                              alpha=alpha)


def mean_rr(rr_list: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([r.float() for r in rr_list], dim=0).mean(0)
