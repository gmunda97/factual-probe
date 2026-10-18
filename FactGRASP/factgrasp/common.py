# bert side of the pipeline from the earlier factual probe code: wikidata split loader, all layer embedding cache, candidate pool and metric helpers

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
COLS = ["subject", "pred_value", "object"]
MAX_LENGTH = 64
EMBED_BATCH = 64

SPLIT_FILES = {
    "train": "wikidata5m_42k_train_relations.csv",
    "valid": "wikidata5m_42k_valid_relations.csv",
    "test":  "wikidata5m_42k_test_relations.csv",
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, SPLIT_FILES[split])
    df = pd.read_csv(path)[COLS]
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


@torch.no_grad()
def _embed_all_layers(
    texts: list[str],
    tokenizer: BertTokenizerFast,
    model: BertModel,
    device: torch.device,
    pooling: str,
) -> torch.Tensor:
    model.eval()
    out_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc=f"embed[{pooling}]", leave=False):
        batch = texts[i : i + EMBED_BATCH]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)
        if pooling == "cls":
            pooled = hs[:, :, 0, :]
        elif pooling == "avg":
            mask = enc["attention_mask"].unsqueeze(1).unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1e-6)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        out_chunks.append(pooled.half().cpu())
    return torch.cat(out_chunks, dim=0)


def _cache_path(split: str, field: str, pooling: str, tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}{split}_{field}_{pooling}_alllayers.pt")


def load_or_compute_field(
    df: pd.DataFrame,
    split: str,
    field: str,
    pooling: str,
    tokenizer: BertTokenizerFast,
    model: BertModel,
    device: torch.device,
    tag: str = "",
) -> torch.Tensor:
    path = _cache_path(split, field, pooling, tag)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    vecs = _embed_all_layers(df[field].tolist(), tokenizer, model, device, pooling)
    torch.save(vecs, path)
    return vecs


@dataclass
class SplitEmb:
    h_s: torch.Tensor
    h_r: torch.Tensor
    h_o: torch.Tensor
    df: pd.DataFrame

    @property
    def num_layers(self) -> int:
        return self.h_s.shape[1]

    def layer(self, l: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.h_s[:, l, :].to(dtype),
            self.h_r[:, l, :].to(dtype),
            self.h_o[:, l, :].to(dtype),
        )


def build_split_embeddings(
    split: str,
    pooling: str,
    tokenizer: BertTokenizerFast,
    model: BertModel,
    device: torch.device,
    max_rows: Optional[int] = None,
    tag: str = "",
) -> SplitEmb:
    df = load_split(split, max_rows=max_rows)
    h_s = load_or_compute_field(df, split, "subject",    pooling, tokenizer, model, device, tag)
    h_r = load_or_compute_field(df, split, "pred_value", pooling, tokenizer, model, device, tag)
    h_o = load_or_compute_field(df, split, "object",     pooling, tokenizer, model, device, tag)
    return SplitEmb(h_s=h_s, h_r=h_r, h_o=h_o, df=df)


def load_bert(device: torch.device) -> tuple[BertTokenizerFast, BertModel]:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


# candidate set = unique object strings of the split in first seen order, 2766 for the wikidata test split
# every model embeds the same strings in its own space
def build_unique_object_pool(
    df: pd.DataFrame, h_o: torch.Tensor
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    names = df["object"].tolist()
    first: dict[str, int] = {}
    for idx, name in enumerate(names):
        if name not in first:
            first[name] = idx
    unique_names = list(first.keys())
    unique_rows = list(first.values())
    unique_h_o = h_o[unique_rows]
    name_to_uid = {name: i for i, name in enumerate(unique_names)}
    true_indices = torch.tensor([name_to_uid[n] for n in names], dtype=torch.long)
    return unique_names, unique_h_o, true_indices


@torch.no_grad()
def compute_metrics(
    queries: torch.Tensor,
    unique_targets: torch.Tensor,
    true_indices: torch.Tensor,
    k_list: tuple = (1, 3, 10),
) -> dict[str, float]:
    q = F.normalize(queries.float(), dim=1)
    t = F.normalize(unique_targets.float(), dim=1)
    true_vecs = unique_targets.float()[true_indices]
    mean_cos = F.cosine_similarity(q, F.normalize(true_vecs, dim=1)).mean().item()
    sim = q @ t.T
    true_scores = sim[torch.arange(len(q)), true_indices].unsqueeze(1)
    ranks = (sim > true_scores).sum(dim=1) + 1
    mrr = (1.0 / ranks.float()).mean().item()
    hits = {f"Hits@{k}": (ranks <= k).float().mean().item() for k in k_list}
    return {"mean_cos_sim": mean_cos, "MRR": mrr, **hits}


@torch.no_grad()
def true_ranks(
    queries: torch.Tensor,
    unique_targets: torch.Tensor,
    true_indices: torch.Tensor,
) -> torch.Tensor:
    q = F.normalize(queries.float(), dim=1)
    t = F.normalize(unique_targets.float(), dim=1)
    sim = q @ t.T
    true_scores = sim[torch.arange(len(q)), true_indices].unsqueeze(1)
    return (sim > true_scores).sum(dim=1) + 1


@torch.no_grad()
def top1_names(
    queries: torch.Tensor, unique_targets: torch.Tensor, unique_names: list[str],
) -> list[str]:
    q = F.normalize(queries.float(), dim=1)
    t = F.normalize(unique_targets.float(), dim=1)
    idx = (q @ t.T).argmax(dim=1).tolist()
    return [unique_names[i] for i in idx]


@torch.no_grad()
def inspect_predictions(
    df: pd.DataFrame,
    queries: torch.Tensor,
    unique_targets: torch.Tensor,
    unique_names: list[str],
    true_indices: torch.Tensor,
    query_indices: Optional[list[int]] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    if query_indices is None:
        query_indices = list(range(min(5, len(df))))
    q = F.normalize(queries.float(), dim=1)
    t = F.normalize(unique_targets.float(), dim=1)
    sim = q @ t.T
    rows = []
    for i in query_indices:
        sims = sim[i]
        true_uid = true_indices[i].item()
        sorted_uids = sims.argsort(descending=True).tolist()
        true_rank = sorted_uids.index(true_uid) + 1
        row = {
            "subject": df["subject"].iloc[i],
            "relation": df["pred_value"].iloc[i],
            "true_object": unique_names[true_uid],
            "true_rank": true_rank,
        }
        for pos, j in enumerate(sorted_uids[:top_k], start=1):
            mark = " (T)" if j == true_uid else ""
            row[f"top_{pos}"] = f"{unique_names[j]}{mark} [{sims[j].item():.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


class LinearTransformation(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_W(
    q_train: torch.Tensor, o_train: torch.Tensor,
    q_val: torch.Tensor, o_val: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
    patience: int = 10, verbose: bool = False,
) -> tuple[LinearTransformation, list[float], list[float]]:
    dim = q_train.shape[1]
    model = LinearTransformation(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    qt, ot = q_train.to(device), o_train.to(device)
    qv, ov = q_val.to(device), o_val.to(device)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_losses, val_losses = [], []
    bad = 0
    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        tr = (1 - F.cosine_similarity(model(qt), ot)).mean()
        tr.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = (1 - F.cosine_similarity(model(qv), ov)).mean()
        sched.step(vl)
        train_losses.append(tr.item())
        val_losses.append(vl.item())
        if vl.item() < best_val:
            best_val = vl.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses


def evaluate_probe_on_split(
    model: torch.nn.Module,
    q: torch.Tensor,
    emb: SplitEmb,
    layer: int,
    device: torch.device,
    apply_W: bool = True,
) -> tuple[dict[str, float], dict[str, float], object]:
    _, _, h_o_l = emb.layer(layer)
    unique_names, unique_h_o, true_idx = build_unique_object_pool(emb.df, h_o_l)
    before = compute_metrics(q, unique_h_o, true_idx)
    with torch.no_grad():
        q_W = model(q.to(device)).cpu() if apply_W else q
    after = compute_metrics(q_W, unique_h_o, true_idx)
    ctx = {
        "df": emb.df, "unique_names": unique_names,
        "unique_h_o": unique_h_o, "true_idx": true_idx,
        "q_before": q, "q_after": q_W,
    }
    return before, after, ctx


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
