# dataset loading and the known valid objects per (s, r) that the filtered ranking needs

from __future__ import annotations

import os
import random
import sys
from typing import Optional, Sequence

import pandas as pd
import torch

from factgrasp import paths

if paths.KG_CODE not in sys.path:
    sys.path.insert(0, paths.KG_CODE)

import common as C

COLS = ["subject", "pred_value", "object"]
# relations held out for the router check in appendix e
HANDPICK_KEYWORDS = ("sport", "team", "league", "member of sports team")
SPLITS = ("train", "valid", "test")
BUILTIN = ("wikidata", "conceptnet")


def load_dataset(name: str, split: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError("split must be one of %s, got %r" % (SPLITS, split))
    if name == "wikidata":
        df = C.load_split(split)
    elif name == "conceptnet":
        df = pd.read_csv(os.path.join(C.DATA_DIR, f"conceptnet_en_{split}_relations.csv"))[COLS]
    else:
        cands = [os.path.join(paths.DATA_EXTRA, "%s_%s_relations.csv" % (name, split)),
                 os.path.join(paths.DATA_EXTRA, "%s_%s.csv" % (name, split))]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            raise FileNotFoundError(
                "unknown dataset %r: expected one of %s"
                % (name, " or ".join(cands)))
        df = pd.read_csv(path)
    df = df[COLS]
    n_before = len(df)
    df = df.dropna(subset=COLS)
    if len(df) < n_before:
        print("[data] %s/%s: dropped %d row(s) with a missing subject, relation or object"
              % (name, split, n_before - len(df)))
    df = df.astype(str).reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def load_all(name: str, max_rows: Optional[int] = None) -> dict:
    return {s: load_dataset(name, s, max_rows) for s in SPLITS}


def concat_splits(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(dfs), ignore_index=True)


# (s, r) -> every known object over train valid and test, feeds the filtered ranking
def valid_sets(df_all: pd.DataFrame) -> dict:
    out: dict = {}
    for s, r, o in zip(df_all["subject"], df_all["pred_value"], df_all["object"]):
        out.setdefault((s, r), set()).add(o)
    return out


def valid_index_lists(df_eval: pd.DataFrame, unique_names: Sequence[str],
                      valid_map: dict) -> list:
    name_to_id = {n: i for i, n in enumerate(unique_names)}
    lists = []
    for s, r, o in zip(df_eval["subject"], df_eval["pred_value"], df_eval["object"]):
        ids = {name_to_id[o]} if o in name_to_id else set()
        for v in valid_map.get((s, r), ()):
            j = name_to_id.get(v)
            if j is not None:
                ids.add(j)
        lists.append(torch.tensor(sorted(ids), dtype=torch.long))
    return lists


def n_valid_in_pool(df_eval: pd.DataFrame, unique_names: Sequence[str],
                    valid_map: dict) -> torch.Tensor:
    lists = valid_index_lists(df_eval, unique_names, valid_map)
    return torch.tensor([max(len(v), 1) for v in lists], dtype=torch.long)


def multiplicity_rate(df_eval: pd.DataFrame, unique_names: Sequence[str],
                      valid_map: dict):
    m = n_valid_in_pool(df_eval, unique_names, valid_map)
    return float((m >= 2).float().mean().item()), m


def unseen_relation_split(df_train: pd.DataFrame, seed: int = 42,
                          frac: float = 0.2) -> set:
    train_rels = sorted(set(df_train["pred_value"]))
    rng = random.Random(seed)
    n_hold = max(1, int(frac * len(train_rels)))
    random_hold = set(rng.sample(train_rels, n_hold))
    handpick = {r for r in train_rels
                if any(k in r.lower() for k in HANDPICK_KEYWORDS)}
    return random_hold | handpick


def cache_tag(dataset: str, enc_tag: str = "", smoke: bool = False) -> str:
    parts = []
    if enc_tag:
        parts.append(enc_tag if enc_tag.endswith("_") else enc_tag + "_")
    if smoke:
        parts.append("smoke_")
    if dataset == "conceptnet":
        parts.append("cn_")
    elif dataset != "wikidata":
        parts.append(dataset + "_")
    return "".join(parts)


def split_name(dataset: str, split: str) -> str:
    return ("cn" + split) if dataset == "conceptnet" else split


def split_embeddings(dataset: str, split: str, pooling: str, tokenizer, model, device,
                     max_rows: Optional[int] = None, enc_tag: str = "",
                     smoke: bool = False):
    df = load_dataset(dataset, split, max_rows)
    tag = cache_tag(dataset, enc_tag, smoke)
    if max_rows is not None and not smoke and not enc_tag:
        raise ValueError("refusing to write a truncated tensor into the full cache: "
                         "pass smoke=True (or an enc_tag) with max_rows")
    stem = split_name(dataset, split)
    h_s = C.load_or_compute_field(df, stem, "subject", pooling, tokenizer, model, device, tag)
    h_r = C.load_or_compute_field(df, stem, "pred_value", pooling, tokenizer, model, device, tag)
    h_o = C.load_or_compute_field(df, stem, "object", pooling, tokenizer, model, device, tag)
    return C.SplitEmb(h_s=h_s, h_r=h_r, h_o=h_o, df=df)
