# small knowledge accounting helpers (bits of mrr) imported by the untrained floor script

from __future__ import annotations

import math

import torch


def recoverable_bits(loss_pool_nats: float, pool_size: int) -> float:
    bits = math.log2(pool_size) - loss_pool_nats / math.log(2.0)
    return float(min(max(bits, 0.0), math.log2(pool_size)))


def bits_added(loss_after_nats: float, loss_before_nats: float, pool_size: int) -> float:
    return recoverable_bits(loss_after_nats, pool_size) - recoverable_bits(loss_before_nats, pool_size)


def effective_rank(W: torch.Tensor) -> float:
    s = torch.linalg.svdvals(W.float())
    p = s / s.sum().clamp_min(1e-12)
    h = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    return float(h.exp())


def verbalisation_split(rr: torch.Tensor, can_say: torch.Tensor) -> dict:
    rr = rr.float(); m = can_say.bool()
    can = rr[m].mean().item() if m.any() else float("nan")
    cannot = rr[~m].mean().item() if (~m).any() else float("nan")
    return {"MRR_can_say": can, "MRR_cannot_say": cannot, "n_can": int(m.sum()),
            "n_cannot": int((~m).sum()), "ratio_cannot_over_can": cannot / can if can else float("nan")}


def fewshot_recovery(mrr_gate: float, mrr_k: float, mrr_ceiling: float) -> float:
    gap = mrr_ceiling - mrr_gate
    return float((mrr_k - mrr_gate) / gap) if gap > 0 else float("nan")
