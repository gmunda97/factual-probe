# the alignment between two models (eq 7) and the relation expertise fit of appendix e (eq 8)

from __future__ import annotations


import torch


# eq 7, closed form ridge from B's composed vectors to A's, no intercept, penalty scaled by n, lambda 1e-2
def fit_stitch(x_src: torch.Tensor, x_dst: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    X = x_src.float(); Y = x_dst.float()
    G = X.t() @ X + ridge * X.shape[0] * torch.eye(X.shape[1])
    return torch.linalg.solve(G, X.t() @ Y)


# for the mismatched pairs control, every row gets paired with a different row
def derangement(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    while True:
        p = torch.randperm(n, generator=g)
        if bool((p != torch.arange(n)).all()):
            return p


def transfer_efficiency(mrr_stitched: float, mrr_native: float) -> float:
    return float(mrr_stitched / mrr_native) if mrr_native > 0 else float("nan")


def stitch_certificate(u: torch.Tensor, v: torch.Tensor, h_o: torch.Tensor,
                       gamma: torch.Tensor) -> torch.Tensor:
    def ang(a, b):
        c = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).clamp(-1, 1)
        return torch.acos(c)
    return ang(u, h_o) + ang(u, v) < gamma.float() / 2


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    xy = (X.t() @ Y).norm() ** 2
    xx = (X.t() @ X).norm(); yy = (Y.t() @ Y).norm()
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


# eq 8, E[m, r] = a_m + b_r + I_mr with relations weighted by their share of test triples
def additive_fit(E: torch.Tensor, w: torch.Tensor):
    E = E.float(); w = w.float() / w.float().sum()
    row = (E * w).sum(1)
    b = (E - row[:, None]).mean(0)
    b = b - (w * b).sum()
    a = row - (w * b).sum()
    I = E - a[:, None] - b[None, :]
    return a, b, I


# share of the weighted variance left in I, 4.1 percent on wikidata
def interaction_share(E: torch.Tensor, w: torch.Tensor) -> float:
    a, b, I = additive_fit(E, w)
    w = w.float() / w.float().sum()
    Ef = E.float(); grand = (Ef * w).sum(1).mean()
    tot = ((Ef - grand) ** 2 * w).sum(1).mean()
    inter = (I ** 2 * w).sum(1).mean()
    return float(inter / tot) if tot > 0 else float("nan")


# oracle that picks the best model for each relation vs the best single model
def routing_gain(E: torch.Tensor, w: torch.Tensor):
    E = E.float(); w = w.float() / w.float().sum()
    m_star = int(torch.argmax((E * w).sum(1)))
    G = float((w * (E.max(0).values - E[m_star])).sum())
    _, _, I = additive_fit(E, w)
    bound = float((w * (I.max(0).values - I[m_star])).sum())
    return G, bound, m_star
