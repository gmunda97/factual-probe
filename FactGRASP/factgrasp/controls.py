# the control baselines of table 8

from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd
import torch


def relation_object_lists(df_train: pd.DataFrame) -> dict:
    out: dict = {}
    for r, o in zip(df_train["pred_value"], df_train["object"]):
        out.setdefault(r, []).append(o)
    return out


def _draw(key, pool: Sequence[str], seed: int) -> str:
    h = hashlib.blake2b(("%d|%s|%s" % (seed, key[0], key[1])).encode("utf-8"),
                        digest_size=8).digest()
    return pool[int.from_bytes(h, "big") % len(pool)]


# random targets: each (s, r) gets one fixed random object of the same relation, scored against the true objects
def control_task_map(df_train: pd.DataFrame, seed: int = 0) -> dict:
    rel_lists = relation_object_lists(df_train)
    all_objs = df_train["object"].tolist()
    out = {}
    for s, r in zip(df_train["subject"], df_train["pred_value"]):
        if (s, r) in out:
            continue
        pool = rel_lists.get(r) or all_objs
        out[(s, r)] = _draw((s, r), pool, seed)
    return out


def apply_control_task(df: pd.DataFrame, rel_lists: dict, fallback: Sequence[str],
                       seed: int = 0) -> pd.DataFrame:
    objs = []
    for s, r in zip(df["subject"], df["pred_value"]):
        pool = rel_lists.get(r) or list(fallback)
        objs.append(_draw((s, r), pool, seed))
    out = df.copy()
    out["object"] = objs
    return out.reset_index(drop=True)


def control_task(df_train: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rel_lists = relation_object_lists(df_train)
    return apply_control_task(df_train, rel_lists, df_train["object"].tolist(), seed)


def control_targets(df: pd.DataFrame, df_train: pd.DataFrame, h_o_train: torch.Tensor,
                    seed: int = 0):
    rel_lists = relation_object_lists(df_train)
    df_c = apply_control_task(df, rel_lists, df_train["object"].tolist(), seed)
    first = {}
    for i, o in enumerate(df_train["object"]):
        if o not in first:
            first[o] = i
    idx = torch.tensor([first[o] for o in df_c["object"]], dtype=torch.long)
    return df_c, h_o_train[idx]


# random encoder: same architecture, random weights
def random_init_encoder(model_name: str, seed: int = 0, device=None):
    import torch as _torch
    from transformers import AutoConfig, AutoTokenizer

    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    lname = model_name.lower()
    if "bert" in lname and "modernbert" not in lname:
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig.from_pretrained(model_name))
    else:
        from transformers import AutoModel, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_name)
        try:
            model = AutoModelForCausalLM.from_config(cfg)
        except (ValueError, TypeError):
            model = AutoModel.from_config(cfg)
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    return tok, model


# majority object: no encoder, objects ranked by how often they answer that relation in training
def majority_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      unique_names: Sequence[str]) -> torch.Tensor:
    names = list(unique_names)
    name_to_id = {n: i for i, n in enumerate(names)}
    U = len(names)
    glob = torch.zeros(U)
    per_rel: dict = {}
    for r, o in zip(df_train["pred_value"], df_train["object"]):
        j = name_to_id.get(o)
        if j is None:
            continue
        glob[j] += 1.0
        vec = per_rel.get(r)
        if vec is None:
            vec = per_rel[r] = torch.zeros(U)
        vec[j] += 1.0
    tie = glob / (glob.sum() + 1.0)
    zero = torch.zeros(U)
    rows = [per_rel.get(r, zero) + tie for r in df_test["pred_value"]]
    return torch.stack(rows, dim=0) if rows else torch.zeros(0, U)


# character trigrams: tf idf cosine between the subject string and each candidate string
def lexical_baseline(df_test: pd.DataFrame, unique_names: Sequence[str],
                     ngram: int = 3, use_relation: bool = False) -> torch.Tensor:
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = [str(n) for n in unique_names]
    if use_relation:
        queries = [("%s %s" % (s, r)) for s, r in zip(df_test["subject"], df_test["pred_value"])]
    else:
        queries = [str(s) for s in df_test["subject"]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram, ngram), min_df=1)
    vec.fit(names + queries)
    Q = vec.transform(queries)
    T = vec.transform(names)
    S = (Q @ T.T).toarray()
    return torch.from_numpy(S).float()


def selectivity(probe_rr: torch.Tensor, control_rr: torch.Tensor, n_boot: int = 2000,
                seed: int = 0) -> dict:
    from factgrasp import metrics
    out = metrics.paired_ci(probe_rr, control_rr, n_boot=n_boot, seed=seed)
    out["probe_MRR"] = float(probe_rr.float().mean().item())
    out["control_MRR"] = float(control_rr.float().mean().item())
    return out
