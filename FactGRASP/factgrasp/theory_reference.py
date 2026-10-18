# reference rank code and a softmax bound the ladder reports next to mrr, metrics.py uses ranks_from_scores and softmax_bound from here

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def ranks_from_scores(S: torch.Tensor, gold: torch.Tensor,
                      valid: Optional[Sequence[torch.Tensor]] = None,
                      tie: str = "optimistic") -> torch.Tensor:
    S = S.clone().float()
    N = S.shape[0]
    if valid is not None:
        for i in range(N):
            others = valid[i][valid[i] != gold[i]]
            if len(others):
                S[i, others] = -float("inf")
    g = S[torch.arange(N), gold].unsqueeze(1)
    greater = (S > g).sum(1)
    if tie == "optimistic":
        return greater + 1
    equal = (S == g).sum(1) - 1
    if tie == "pessimistic":
        return greater + equal + 1
    if tie == "mean":
        return greater + 1 + equal.float() / 2
    raise ValueError(tie)


def mrr_hits(ranks: torch.Tensor, ks=(1, 3, 10)) -> dict:
    r = ranks.float()
    out = {"MRR": (1.0 / r).mean().item()}
    out.update({f"Hits@{k}": (r <= k).float().mean().item() for k in ks})
    return out


def softmax_bound(S: torch.Tensor, gold: torch.Tensor, tau: float,
                  valid: Optional[Sequence[torch.Tensor]] = None) -> dict:
    S = S.clone().float()
    N = S.shape[0]
    if valid is not None:
        for i in range(N):
            others = valid[i][valid[i] != gold[i]]
            if len(others):
                S[i, others] = -float("inf")
    logq = F.log_softmax(S / tau, dim=1)[torch.arange(N), gold]
    q = logq.exp()
    rr = 1.0 / ranks_from_scores(S, gold, None, "optimistic").float()
    return {"B_tau": q.mean().item(), "exp_neg_L": logq.mean().exp().item(),
            "MRR": rr.mean().item(), "gap": (rr - q).mean().item(),
            "violations": int((q > rr + 1e-6).sum())}


def pool_margin(pool: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    P = F.normalize(pool.float(), dim=1)
    sim = P @ P.t()
    sim.fill_diagonal_(-2.0)
    nn_cos = sim.max(dim=1).values.clamp(-1, 1)
    return torch.arccos(nn_cos)[gold]


def cosine_markov_bound(cos_to_gold: torch.Tensor, margin: torch.Tensor) -> float:
    ang = torch.arccos(cos_to_gold.clamp(-1, 1))
    return (ang < margin / 2).float().mean().item()


def ambiguity_ceiling(n_valid_in_pool: torch.Tensor) -> float:
    m = n_valid_in_pool.float()
    H = torch.tensor([sum(1.0 / j for j in range(1, int(v) + 1)) for v in m])
    return (H / m).mean().item()


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += (1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else
                  -1 if (a[i] - a[j]) * (b[i] - b[j]) < 0 else 0)
    return 2.0 * s / (n * (n - 1)) if n > 1 else float("nan")


def relation_distance(h_rel: torch.Tensor, metric: str = "cosine",
                      G: Optional[torch.Tensor] = None) -> torch.Tensor:
    if metric == "cosine":
        Z = F.normalize(h_rel.float(), dim=1)
        return (1.0 - Z @ Z.t()).clamp_min(0)
    if metric == "gate":
        Z = h_rel.float() @ G.t()
        return torch.cdist(Z, Z)
    raise ValueError(metric)


def nearest_seen(D: torch.Tensor, seen: torch.Tensor, unseen: torch.Tensor):
    d = D[unseen][:, seen].min(dim=1).values
    return d, d.max().item(), d.mean().item()


def transfer_modulus(T: torch.Tensor, D: torch.Tensor, seen: torch.Tensor,
                     quantile: float = 0.9) -> float:
    idx = seen
    Tsub, Dsub = T[idx][:, idx], D[idx][:, idx]
    mask = ~torch.eye(len(idx), dtype=torch.bool) & (Dsub > 1e-6)
    ratio = (Tsub[mask].clamp_min(0) / Dsub[mask])
    return torch.quantile(ratio, quantile).item()


def expert_modulus(risk_seen: torch.Tensor, D: torch.Tensor, seen: torch.Tensor,
                   quantile: float = 0.9) -> float:
    R = risk_seen.float().cpu()
    Dsub = D[seen][:, seen].float().cpu()
    n = len(seen)
    mask = ~torch.eye(n, dtype=torch.bool) & (Dsub > 1e-6)
    diff = (R.unsqueeze(0) - R.unsqueeze(1)).clamp_min(0)
    ratio = (diff / Dsub.unsqueeze(-1))[mask]
    ratio = ratio[torch.isfinite(ratio)]
    if ratio.numel() == 0:
        return float("nan")
    return torch.quantile(ratio, quantile).item()


def routing_decomposition(risk_gate: torch.Tensor, risk_per_expert: torch.Tensor,
                          risk_fewshot: torch.Tensor) -> dict:
    orc = risk_per_expert.min(dim=1).values
    Q = risk_gate - orc
    C = orc - risk_fewshot
    return {"Q": Q, "C": C, "oracle": orc,
            "Q_over_C_mean": (Q.mean() / C.mean().clamp_min(1e-9)).item()}


def effective_experts(g: torch.Tensor) -> dict:
    def ent(p):
        return -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return {"N_eff_global": ent(g.mean(0)).exp().item(),
            "N_eff_per_input": ent(g).exp().mean().item()}


def paired_bootstrap(rr_a: torch.Tensor, rr_b: torch.Tensor, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    g = torch.Generator().manual_seed(seed)
    d = (rr_a - rr_b).float(); N = len(d)
    idx = torch.randint(0, N, (n_boot, N), generator=g)
    means = d[idx].mean(1)
    lo, hi = torch.quantile(means, alpha / 2).item(), torch.quantile(means, 1 - alpha / 2).item()
    return {"diff": d.mean().item(), "ci_low": lo, "ci_high": hi,
            "significant": not (lo <= 0.0 <= hi)}
