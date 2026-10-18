# loss choices for the entity disjoint split run, cosine vs in batch infonce

import sys

import torch

from factgrasp import paths

sys.path.insert(0, paths.KG_CODE)
import variants as V


def margin_loss(scores: torch.Tensor, gold: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    s_o = scores.gather(1, gold.view(-1, 1))
    h = (gamma - (s_o - scores)).clamp_min(0.0) / gamma
    h.scatter_(1, gold.view(-1, 1), 0.0)
    return h.sum(1)


def make_objective(name: str, temp: float, gamma: float, pool_targets=None):
    if name == "squared_error":
        def f(oh, o):
            return ((oh - torch.nn.functional.normalize(o, dim=-1)) ** 2).sum(-1).mean()
        return f
    if name == "squared_error_normalised":
        def f(oh, o):
            return ((torch.nn.functional.normalize(oh, dim=-1)
                     - torch.nn.functional.normalize(o, dim=-1)) ** 2).sum(-1).mean()
        return f
    if name == "cosine":
        return lambda oh, o: V.cosine_loss(oh, o)
    if name == "margin":
        def f(oh, o):
            q = torch.nn.functional.normalize(oh, dim=-1)
            t = torch.nn.functional.normalize(o, dim=-1)
            scores = q @ t.t()
            gold = torch.arange(scores.shape[0], device=scores.device)
            return margin_loss(scores, gold, gamma=gamma).mean()
        return f
    if name == "infonce_batch":
        return lambda oh, o: V.infonce_loss(oh, o, temp)
    if name == "infonce_pooled":
        def f(oh, o, _P=pool_targets):
            q = torch.nn.functional.normalize(oh, dim=-1)
            t = torch.nn.functional.normalize(o, dim=-1)
            P = torch.nn.functional.normalize(_P.to(q.device), dim=-1)
            pos = (q * t).sum(-1, keepdim=True) / temp
            neg = q @ P.t() / temp
            return -(pos.squeeze(1) - torch.logsumexp(
                torch.cat([pos, neg], dim=1), dim=1)).mean()
        return f
    raise ValueError(name)
