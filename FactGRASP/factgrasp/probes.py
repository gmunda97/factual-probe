# map builders and the minibatch trainer used by the ladder

from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from factgrasp import paths
sys.path.insert(0, paths.KG_CODE)
import variants as V


# infonce where other correct objects of the same (s, r) in the batch are masked so a second right answer is not pushed away
def masked_infonce_loss(o_hat, o, valid_mask, temp=0.05):
    q = F.normalize(o_hat, dim=1); k = F.normalize(o, dim=1)
    logits = (q @ k.t()) / temp
    eye = torch.eye(len(q), dtype=torch.bool, device=q.device)
    logits = logits.masked_fill(valid_mask & ~eye, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


def pool_infonce_loss(o_hat, gold_idx, bank, temp=0.05):
    q = F.normalize(o_hat, dim=1); k = F.normalize(bank, dim=1)
    return F.cross_entropy((q @ k.t()) / temp, gold_idx)


class LowRankMixtureOfW(nn.Module):
    def __init__(self, dim, num_experts, gate_dim, rank=128, gate_temp=1.0):
        super().__init__()
        self.E = num_experts
        self.U = nn.Parameter(0.01 * torch.randn(num_experts, dim, rank))
        self.Vt = nn.Parameter(0.01 * torch.randn(num_experts, rank, dim))
        self.gate = nn.Linear(gate_dim, num_experts); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.gate_temp = gate_temp
    def gate_weights(self, h_r):
        return F.softmax(self.gate(h_r) / self.gate_temp, dim=-1)
    def forward(self, x, h_r):
        g = self.gate_weights(h_r)
        low = torch.einsum("nd,erd->ner", x, self.Vt)
        outs = torch.einsum("ner,edr->ned", low, self.U) + x.unsqueeze(1)
        return (g.unsqueeze(-1) * outs).sum(1), g
    @torch.no_grad()
    def forward_single_expert(self, x, e):
        return x + (x @ self.Vt[e].t()) @ self.U[e].t()


def make_probe(D, E=1, rank=None, gate_type="softmax", proto_init=None, use_shared=False):
    if rank is None:
        return V.FlexMixtureOfW(D, D, E, gate_dim=D, gate_type=gate_type,
                                proto_init=proto_init, use_shared=use_shared)
    return LowRankMixtureOfW(D, E, gate_dim=D, rank=rank)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def train_minibatch_idx(model, x_tr, hr_tr, o_tr, x_va, hr_va, o_va, device, loss_fn_idx,
                        val_loss_fn, num_epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=512,
                        lambda_bal=0.01, patience=12, is_mixture=True, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    N = x_tr.shape[0]
    xt, rt, ot = x_tr.to(device), hr_tr.to(device), o_tr.to(device)
    xv, rv, ov = x_va.to(device), hr_va.to(device), o_va.to(device)
    best = float("inf"); best_state = V._clone_state(model); bad = 0
    for _ in range(num_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(xt[idx], rt[idx]); o_hat, g = out if isinstance(out, tuple) else (out, None)
            loss = loss_fn_idx(o_hat, ot[idx], idx)
            if is_mixture and g is not None:
                loss = loss + lambda_bal * V.load_balance_loss(g)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xv, rv); o_hat_v = out[0] if isinstance(out, tuple) else out
            vl = val_loss_fn(o_hat_v, ov).item()
        sched.step(vl)
        if vl < best: best, bad, best_state = vl, 0, V._clone_state(model)
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model
