# per layer pca, the composed vector and the layer sweep used to pick each model's layer

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from typing import Iterable, Optional, Sequence

import torch

from factgrasp import paths
from factgrasp import data as D
from factgrasp import metrics as M
from factgrasp.encoders import PCAProjector

sys.path.insert(0, paths.KG_CODE)
sys.path.insert(0, paths.LOES_ROOT)
import common as C
import variants as V
from loes import select_layers_from_custom_embeddings

FIELDS = ("subject", "pred_value", "object")


@dataclass
class SweepCache:
    train: dict
    valid: dict
    df_train: object
    df_valid: object
    valid_map: Optional[dict] = None
    pca: dict = dc_field(default_factory=dict)

    def layers(self) -> list[int]:
        return sorted(set(self.train[FIELDS[0]]).intersection(self.valid[FIELDS[0]]))


def _as_cache(cache) -> SweepCache:
    if isinstance(cache, SweepCache):
        return cache
    return SweepCache(**cache)


def fit_projector(cache: SweepCache, layer: int, D_target: int = 768,
                  seed: int = 0) -> Optional[PCAProjector]:
    rows = torch.cat([cache.train[f][layer].float() for f in FIELDS], dim=0)
    if D_target is None or rows.shape[1] <= D_target:
        return None
    return PCAProjector.fit(rows, k=D_target, seed=seed)


def project_split(split: dict, layer: int, proj: Optional[PCAProjector]) -> tuple:
    out = []
    for f in FIELDS:
        X = split[f][layer].float()
        out.append(X if proj is None else proj.transform(X))
    return tuple(out)


# x = h_s + h_r (eq 1)
def compose_query(h_s: torch.Tensor, h_r: torch.Tensor, mode: str = "add") -> torch.Tensor:
    return V.compose(h_s, h_r, mode)


def _pool_and_gold(df, h_o: torch.Tensor, valid_map: Optional[dict]):
    unique_names, pool, gold = C.build_unique_object_pool(df, h_o)
    valid_lists = None
    if valid_map is not None:
        valid_lists = D.valid_index_lists(df, unique_names, valid_map)
    return unique_names, pool, gold, valid_lists


def train_single_map(x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device, loss="infonce",
                     epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=512,
                     temp=0.05, seed=0, num_experts=1, gate_type="softmax",
                     patience=12):
    C.set_seed(seed)
    dim = x_tr.shape[1]
    model = V.FlexMixtureOfW(dim, o_tr.shape[1], num_experts, gate_dim=hr_tr.shape[1],
                             gate_type=gate_type).to(device)
    loss_fn = V.make_loss(loss, temp=temp)
    model = V.train_minibatch(model, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device,
                              loss_fn, num_epochs=epochs, lr=lr,
                              weight_decay=weight_decay, batch_size=batch_size,
                              patience=patience, is_mixture=num_experts > 1)
    return model


@torch.no_grad()
def predict(model, x, hr, device, batch=8192) -> torch.Tensor:
    outs, gates = [], []
    for i in range(0, x.shape[0], batch):
        out = model(x[i:i + batch].to(device), hr[i:i + batch].to(device))
        o_hat, g = out if isinstance(out, tuple) else (out, None)
        outs.append(o_hat.float().cpu())
        if g is not None:
            gates.append(g.float().cpu())
    return torch.cat(outs, 0), (torch.cat(gates, 0) if gates else None)


# layer selection: a linear map with cosine+infonce at every candidate layer scored on the valid split,
# the paper uses the layer with the best valid mrr
def sweep(cache, layers: Iterable[int], D_target: int = 768,
          seeds: Sequence[int] = (0,), loss: str = "infonce", device=None,
          epochs: int = 60, batch_size: int = 512, temp: float = 0.05,
          tie: str = "random", return_details: bool = False) -> dict:
    cache = _as_cache(cache)
    device = device or torch.device("cpu")
    out, details = {}, {}
    for layer in sorted(set(int(l) for l in layers)):
        proj = cache.pca.get(layer)
        if proj is None:
            proj = fit_projector(cache, layer, D_target, seed=0)
            cache.pca[layer] = proj
        hs_tr, hr_tr, ho_tr = project_split(cache.train, layer, proj)
        hs_va, hr_va, ho_va = project_split(cache.valid, layer, proj)
        x_tr, x_va = compose_query(hs_tr, hr_tr), compose_query(hs_va, hr_va)
        _, pool, gold, valid_lists = _pool_and_gold(cache.df_valid, ho_va, cache.valid_map)

        per_seed = []
        for s in seeds:
            model = train_single_map(x_tr, hr_tr, ho_tr, x_va, hr_va, ho_va, device,
                                     loss=loss, epochs=epochs, batch_size=batch_size,
                                     temp=temp, seed=s)
            o_hat, _ = predict(model, x_va, hr_va, device)
            res = M.evaluate(o_hat, pool, gold, valid=valid_lists, tie=tie)
            per_seed.append(res)
        out[layer] = float(sum(r["MRR"] for r in per_seed) / len(per_seed))
        details[layer] = {k: float(sum(r[k] for r in per_seed) / len(per_seed))
                          for k in per_seed[0]
                          if isinstance(per_seed[0][k], (int, float))
                          and not isinstance(per_seed[0][k], bool)}
    if return_details:
        return out, details
    return out


def loes_select(cache_valid: torch.Tensor, targets: Optional[torch.Tensor] = None,
                k: int = 3, layers: Optional[Sequence[int]] = None,
                alpha_label_free: float = 1.0, alpha_supervised: float = 0.0,
                targets_per_layer: Optional[Sequence[torch.Tensor]] = None,
                reg: float = 1e-3) -> dict:
    if cache_valid.ndim != 3:
        raise ValueError("cache_valid must be (n, L_cand, D)")
    n_cand = cache_valid.shape[1]
    if layers is None:
        layers = list(range(n_cand))
    layers = list(layers)
    if len(layers) != n_cand:
        raise ValueError("len(layers) must match cache_valid.shape[1]")

    def remap(idxs):
        return [int(layers[i]) for i in idxs]

    X = cache_valid.float()
    res_lf = select_layers_from_custom_embeddings(embeddings=X, k=k, task="label_free",
                                                  alpha=alpha_label_free)
    out = {
        "label_free": remap(res_lf.selected_layers),
        "label_free_scores": [float(s) for s in res_lf.layer_scores],
        "supervised": None,
        "supervised_scores": None,
        "supervised_same_layer": None,
        "supervised_same_layer_scores": None,
        "candidate_layers": layers,
    }

    if targets is not None:
        res_sup = select_layers_from_custom_embeddings(
            embeddings=X, targets=targets.float(), k=k, task="regression",
            alpha=alpha_supervised, reg=reg)
        out["supervised"] = remap(res_sup.selected_layers)
        out["supervised_scores"] = [float(s) for s in res_sup.layer_scores]

    if targets_per_layer is not None:
        if len(targets_per_layer) != n_cand:
            raise ValueError("targets_per_layer must have one target per candidate layer")
        scores = []
        for i in range(n_cand):
            r = select_layers_from_custom_embeddings(
                embeddings=X[:, i:i + 1, :].contiguous(),
                targets=targets_per_layer[i].float(), k=1, task="regression",
                alpha=alpha_supervised, reg=reg)
            scores.append(float(r.layer_scores[0]))
        order = sorted(range(n_cand), key=lambda i: scores[i])[:k]
        out["supervised_same_layer"] = remap(order)
        out["supervised_same_layer_scores"] = scores
    return out


def relative_depth(best: int, L: int) -> float:
    if L <= 0:
        raise ValueError("L must be positive")
    return float(best) / float(L)


def selector_regret(mrr_by_layer: dict, selected_layer: int) -> float:
    if not mrr_by_layer:
        raise ValueError("mrr_by_layer is empty")
    keyed = {int(k): float(v) for k, v in mrr_by_layer.items()}
    if int(selected_layer) not in keyed:
        raise KeyError(
            f"layer {selected_layer} was not swept; sweep it before asking for regret")
    return max(keyed.values()) - keyed[int(selected_layer)]


def best_layer(mrr_by_layer: dict) -> int:
    keyed = {int(k): float(v) for k, v in mrr_by_layer.items()}
    return max(keyed, key=keyed.get)


def depth_report(mrr_by_layer: dict, L: int, loes_label_free_top1: Optional[int] = None,
                 loes_supervised_top1: Optional[int] = None) -> dict:
    keyed = {int(k): float(v) for k, v in mrr_by_layer.items()}
    l_star = best_layer(keyed)
    mu_star = keyed[l_star]
    mu_last = keyed.get(int(L))
    rel_drop_last = None
    if mu_last is not None and mu_star > 0:
        rel_drop_last = (mu_star - mu_last) / mu_star
    out = {
        "best_layer": l_star,
        "best_mrr": mu_star,
        "last_layer": int(L),
        "last_layer_mrr": mu_last,
        "lambda_star": relative_depth(l_star, L),
        "last_layer_relative_drop": rel_drop_last,
        "mrr_by_layer": keyed,
    }
    if loes_label_free_top1 is not None:
        out["loes_label_free_top1"] = int(loes_label_free_top1)
        try:
            out["selector_regret_label_free"] = selector_regret(keyed, loes_label_free_top1)
        except KeyError:
            out["selector_regret_label_free"] = None
    if loes_supervised_top1 is not None:
        out["loes_supervised_top1"] = int(loes_supervised_top1)
        try:
            out["selector_regret_supervised"] = selector_regret(keyed, loes_supervised_top1)
        except KeyError:
            out["selector_regret_supervised"] = None
    return out
