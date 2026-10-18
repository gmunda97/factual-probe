# the map itself (single linear map and the mixture of transformations) and the losses

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compose(hs: torch.Tensor, hr: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "add":
        return hs + hr
    if mode == "concat":
        return torch.cat([hs, hr], dim=-1)
    raise ValueError(f"unknown compose mode {mode!r}")


def compose_dim(d: int, mode: str) -> int:
    return d if mode == "add" else 2 * d


# this is the map. num_experts=1 gives the single linear map W (eq 2), 8 gives the MoT (eq 3)
# 8 matrices W_i mixed by g = softmax(G h_r + b) (eq 4), the gate only sees the relation vector
# experts start at identity plus noise and the gate at zero so all 8 start equal
class FlexMixtureOfW(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_experts: int,
                 gate_dim: int, gate_temp: float = 1.0,
                 gate_type: str = "softmax",
                 proto_init: Optional[torch.Tensor] = None,
                 use_shared: bool = False) -> None:
        super().__init__()
        self.E = num_experts
        self.gate_type = gate_type
        self.gate_temp = gate_temp
        self.use_shared = use_shared

        if use_shared:
            base = torch.eye(out_dim, in_dim) if in_dim == out_dim else 0.02 * torch.randn(out_dim, in_dim)
            self.shared = nn.Parameter(base + 0.01 * torch.randn(out_dim, in_dim))

        experts = []
        for _ in range(num_experts):
            if use_shared:
                experts.append(0.01 * torch.randn(out_dim, in_dim))
            elif in_dim == out_dim:
                experts.append(torch.eye(out_dim) + 0.01 * torch.randn(out_dim, in_dim))
            else:
                experts.append(0.02 * torch.randn(out_dim, in_dim))
        self.experts = nn.Parameter(torch.stack(experts))

        if gate_type == "softmax":
            self.gate = nn.Linear(gate_dim, num_experts, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif gate_type == "fuzzy":
            if proto_init is not None:
                assert proto_init.shape == (num_experts, gate_dim)
                self.protos = nn.Parameter(proto_init.clone().float())
            else:
                self.protos = nn.Parameter(0.1 * torch.randn(num_experts, gate_dim))
            self.log_sigma = nn.Parameter(torch.tensor(-1.0))
        else:
            raise ValueError(f"unknown gate_type {gate_type!r}")

    def gate_weights(self, h_r: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "softmax":
            return F.softmax(self.gate(h_r) / self.gate_temp, dim=-1)
        sigma = F.softplus(self.log_sigma) + 1e-3
        hn = F.normalize(h_r, dim=-1)
        pn = F.normalize(self.protos, dim=-1)
        d2 = torch.cdist(hn, pn) ** 2
        return F.softmax(-d2 / (2 * sigma ** 2), dim=-1)

    def forward(self, x: torch.Tensor, h_r: torch.Tensor):
        g = self.gate_weights(h_r)
        outs = torch.stack([F.linear(x, self.experts[e]) for e in range(self.E)], dim=1)
        o_hat = (g.unsqueeze(-1) * outs).sum(dim=1)
        if self.use_shared:
            o_hat = o_hat + F.linear(x, self.shared)
        return o_hat, g

    @torch.no_grad()
    def forward_single_expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        o = F.linear(x, self.experts[e])
        if self.use_shared:
            o = o + F.linear(x, self.shared)
        return o


# gate balancing penalty sum g_bar log(8 g_bar), weighted 0.01 in training, stops the gate collapsing on one expert
def load_balance_loss(g: torch.Tensor) -> torch.Tensor:
    imp = g.mean(dim=0)
    E = g.shape[1]
    return (imp * torch.log(imp.clamp_min(1e-9) * E)).sum()


# eq 5
def cosine_loss(o_hat: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(o_hat, o)).mean()


# in batch infonce, the other objects in the batch are the negatives, temp 0.05
def infonce_loss(o_hat: torch.Tensor, o: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    q = F.normalize(o_hat, dim=1)
    k = F.normalize(o, dim=1)
    logits = (q @ k.t()) / temp
    labels = torch.arange(len(q), device=q.device)
    return F.cross_entropy(logits, labels)


def softrank_hits_loss(o_hat: torch.Tensor, o: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    q = F.normalize(o_hat, dim=1)
    k = F.normalize(o, dim=1)
    s = q @ k.t()
    diag = s.diagonal().unsqueeze(1)
    gaps = (s - diag) / temp
    eye = torch.eye(len(q), device=q.device).bool()
    soft_rank = torch.sigmoid(gaps).masked_fill(eye, 0.0).sum(dim=1)
    return (soft_rank / max(len(q) - 1, 1)).mean()


def cosface_loss(o_hat: torch.Tensor, o: torch.Tensor,
                 margin: float = 0.2, scale: float = 20.0) -> torch.Tensor:
    q = F.normalize(o_hat, dim=1)
    k = F.normalize(o, dim=1)
    s = q @ k.t()
    eye = torch.eye(len(q), device=q.device).bool()
    s = s - margin * eye.float()
    labels = torch.arange(len(q), device=q.device)
    return F.cross_entropy(scale * s, labels)


def make_loss(name: str, lam: float = 1.0, temp: float = 0.05,
              margin: float = 0.2, scale: float = 20.0):
    if name == "cosine":
        return lambda oh, o: cosine_loss(oh, o)
    # the paper's default objective: cosine plus infonce with equal weight (eq 6)
    if name == "infonce":
        return lambda oh, o: cosine_loss(oh, o) + lam * infonce_loss(oh, o, temp)
    if name == "softrank":
        return lambda oh, o: cosine_loss(oh, o) + lam * softrank_hits_loss(oh, o, temp)
    if name == "cosface":
        return lambda oh, o: cosine_loss(oh, o) + lam * cosface_loss(oh, o, margin, scale)
    if name == "infonce_only":
        return lambda oh, o: infonce_loss(oh, o, temp)
    raise ValueError(f"unknown loss {name!r}")


def _clone_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def train_fullbatch(model, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device,
                    num_epochs=300, lr=1e-3, weight_decay=1e-4,
                    lambda_bal=0.01, patience=20, is_mixture=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    xt, rt, ot = x_tr.to(device), hr_tr.to(device), o_tr.to(device)
    xv, rv, ov = x_va.to(device), hr_va.to(device), o_va.to(device)
    best = float("inf"); best_state = _clone_state(model); bad = 0
    for _ in range(num_epochs):
        model.train(); opt.zero_grad()
        out = model(xt, rt)
        o_hat, g = out if isinstance(out, tuple) else (out, None)
        loss = cosine_loss(o_hat, ot)
        if is_mixture and g is not None:
            loss = loss + lambda_bal * load_balance_loss(g)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xv, rv)
            o_hat_v = out[0] if isinstance(out, tuple) else out
            vl = cosine_loss(o_hat_v, ov).item()
        sched.step(vl)
        if vl < best:
            best = vl; bad = 0; best_state = _clone_state(model)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


# adamw lr 1e-3 wd 1e-4 batch 512 up to 60 epochs, lr halves after 4 bad val checks, stops after 12
def train_minibatch(model, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device,
                    loss_fn, num_epochs=60, lr=1e-3, weight_decay=1e-4,
                    batch_size=512, lambda_bal=0.01, patience=12, is_mixture=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    N = x_tr.shape[0]
    xt, rt, ot = x_tr.to(device), hr_tr.to(device), o_tr.to(device)
    xv, rv, ov = x_va.to(device), hr_va.to(device), o_va.to(device)
    best = float("inf"); best_state = _clone_state(model); bad = 0
    for _ in range(num_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(xt[idx], rt[idx])
            o_hat, g = out if isinstance(out, tuple) else (out, None)
            loss = loss_fn(o_hat, ot[idx])
            if is_mixture and g is not None:
                loss = loss + lambda_bal * load_balance_loss(g)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xv, rv)
            o_hat_v = out[0] if isinstance(out, tuple) else out
            vl = loss_fn(o_hat_v, ov).item()
        sched.step(vl)
        if vl < best:
            best = vl; bad = 0; best_state = _clone_state(model)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model
