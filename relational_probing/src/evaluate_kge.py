"""
Evaluate the KGE-target relational-probing linear transformation on the test set.

h_o is taken from the pre-computed KGE obj_embedding column in the CSV.
h_s and h_r are encoded by the language model.
W maps: LM space (D_in) → KGE space (D_out).

Metrics reported (after applying W):
  - Mean cosine similarity
  - Mean Reciprocal Rank  (MRR)
  - Hits@k                for k in {1, 3, 10}

Usage:
    python evaluate_kge.py
"""

import os
import sys
from datetime import datetime
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS
from transformations import LinearTransformation
from config import get_config

COLS = ["subject", "pred_value", "object", "obj_embedding"]


def _parse_kge_embeddings(df: pd.DataFrame) -> torch.Tensor:
    """Convert the space-separated obj_embedding strings to a (N, D_kge) tensor."""
    return torch.tensor(
        df["obj_embedding"].apply(lambda s: list(map(float, s.split()))).tolist(),
        dtype=torch.float32,
    )


class KGEEvaluator:
    def __init__(
        self,
        data_path: str,
        cache_dir: str,
        model_path: str,
        model_name: str = "bert-base-uncased",
        cache_prefix: str = "42k_test_rel",
        k_list: tuple = (1, 3, 10),
    ) -> None:
        self.data_path    = data_path
        self.cache_dir    = cache_dir
        self.model_path   = model_path
        self.model_name   = model_name
        self.cache_prefix = cache_prefix
        self.k_list       = k_list
        self._encoder: Optional[BERTEmbeddingsWithCLS] = None

    @property
    def encoder(self) -> BERTEmbeddingsWithCLS:
        if self._encoder is None:
            print(f"Loading encoder: {self.model_name}")
            self._encoder = BERTEmbeddingsWithCLS(self.model_name)
        return self._encoder

    def _embed_strings(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        enc = self.encoder
        enc.model.eval()
        all_vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch = texts[i : i + batch_size]
            inputs = enc.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            with torch.no_grad():
                out = enc.model(**inputs)
            all_vecs.append(enc.pool(out.last_hidden_state, inputs["attention_mask"]))
        return torch.cat(all_vecs, dim=0)

    def _load_or_compute(self, texts: list[str], cache_path: str) -> torch.Tensor:
        if os.path.exists(cache_path):
            print(f"  [cache hit]  {os.path.basename(cache_path)}")
            return torch.load(cache_path, map_location="cpu")
        print(f"  [computing]  {os.path.basename(cache_path)}")
        vecs = self._embed_strings(texts)
        torch.save(vecs, cache_path)
        return vecs

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)[COLS]
        print(f"Test set: {len(df):,} triples  ({self.data_path})")
        return df

    def load_embeddings(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h_s, h_r, h_o) where h_o comes from the KGE column."""
        h_s = self._load_or_compute(
            df["subject"].tolist(),
            os.path.join(self.cache_dir, f"{self.cache_prefix}_h_s_bert_cls_kge.pt"),
        )
        h_r = self._load_or_compute(
            df["pred_value"].tolist(),
            os.path.join(self.cache_dir, f"{self.cache_prefix}_h_r_bert_cls_kge.pt"),
        )
        h_o = _parse_kge_embeddings(df)
        print(f"  h_s: {h_s.shape}  h_r: {h_r.shape}  h_o (KGE): {h_o.shape}")
        return h_s, h_r, h_o

    def load_model(self, d_in: int, d_out: int) -> LinearTransformation:
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        model = LinearTransformation(d_in, d_out)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Loaded model from {self.model_path}  ({d_in} → {d_out})")
        return model

    def build_unique_object_pool(
        self, df: pd.DataFrame, h_o: torch.Tensor
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Deduplicate the KGE object pool.

        Returns unique_names, unique_h_o (U, D_kge), true_indices (N,).
        """
        object_names = df["object"].tolist()
        first_occurrence: dict[str, int] = {}
        for idx, name in enumerate(object_names):
            if name not in first_occurrence:
                first_occurrence[name] = idx

        unique_names   = list(first_occurrence.keys())
        unique_row_ids = list(first_occurrence.values())
        unique_h_o     = h_o[unique_row_ids]

        name_to_uid = {name: i for i, name in enumerate(unique_names)}
        true_indices = torch.tensor(
            [name_to_uid[name] for name in object_names], dtype=torch.long
        )

        print(f"  Object pool: {len(object_names):,} triples → {len(unique_names):,} unique objects")
        return unique_names, unique_h_o, true_indices

    def compute_metrics(
        self,
        queries: torch.Tensor,
        unique_targets: torch.Tensor,
        true_indices: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            q_norm = F.normalize(queries, dim=1)
            t_norm = F.normalize(unique_targets, dim=1)

            true_vecs = unique_targets[true_indices]
            mean_cos  = F.cosine_similarity(
                F.normalize(queries, dim=1),
                F.normalize(true_vecs, dim=1),
            ).mean().item()

            sim_matrix  = q_norm @ t_norm.T
            true_scores = sim_matrix[
                torch.arange(len(queries)), true_indices
            ].unsqueeze(1)
            ranks = (sim_matrix > true_scores).sum(dim=1) + 1

            mrr    = (1.0 / ranks.float()).mean().item()
            hits_k = {k: (ranks <= k).float().mean().item() for k in self.k_list}

        return {
            "mean_cos_sim": mean_cos,
            "MRR": mrr,
            **{f"Hits@{k}": v for k, v in hits_k.items()},
        }

    def inspect_predictions(
        self,
        df: pd.DataFrame,
        queries: torch.Tensor,
        unique_targets: torch.Tensor,
        unique_names: list[str],
        true_indices: torch.Tensor,
        query_indices: Optional[list[int]] = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if query_indices is None:
            query_indices = list(range(min(5, len(df))))

        rows = []
        with torch.no_grad():
            q_norm = F.normalize(queries, dim=1)
            t_norm = F.normalize(unique_targets, dim=1)
            sim_matrix = q_norm @ t_norm.T

        for i in query_indices:
            sims     = sim_matrix[i]
            true_uid = true_indices[i].item()
            true_sim = sims[true_uid].item()

            sorted_uids = sims.argsort(descending=True).tolist()
            true_rank   = sorted_uids.index(true_uid) + 1

            top_candidates = [
                (unique_names[j], round(sims[j].item(), 4))
                for j in sorted_uids[:top_k]
            ]

            row = {
                "query_idx":   i,
                "subject":     df["subject"].iloc[i],
                "relation":    df["pred_value"].iloc[i],
                "true_object": unique_names[true_uid],
                "true_rank":   true_rank,
                "true_sim":    round(true_sim, 4),
            }
            for rank_pos, (name, sim) in enumerate(top_candidates, start=1):
                is_true = " ✓" if name == unique_names[true_uid] else ""
                row[f"top_{rank_pos}"] = f"{name}{is_true} ({sim})"

            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(
        self,
        metrics_df: pd.DataFrame,
        inspection: pd.DataFrame,
        output_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            metrics_df.to_excel(writer, sheet_name="Metrics")
            inspection.to_excel(writer, sheet_name="Inspection After W", index=False)
        print(f"Results saved → {output_path}")


if __name__ == "__main__":
    config = get_config()

    evaluator = KGEEvaluator(
        data_path=config["data_paths"]["test"],
        cache_dir=os.path.join(_ROOT, "data", "embeddings", "relational_probing"),
        model_path=config["model_paths"]["saved_model"],
        model_name=config["model_name"],
    )

    df_test      = evaluator.load_data()
    h_s, h_r, h_o = evaluator.load_embeddings(df_test)
    h_o_hat      = h_s + h_r   # (N, D_in)

    unique_names, unique_h_o, true_indices = evaluator.build_unique_object_pool(df_test, h_o)

    model = evaluator.load_model(d_in=h_o_hat.shape[1], d_out=h_o.shape[1])
    with torch.no_grad():
        h_o_hat_W = model(h_o_hat)   # (N, D_out)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    metrics = evaluator.compute_metrics(h_o_hat_W, unique_h_o, true_indices)
    metrics_df = pd.DataFrame({"After W": metrics}).T.round(4)
    print("\n=== Evaluation results ===")
    print(metrics_df.to_string())

    # ── Per-query inspection ─────────────────────────────────────────────────
    print("\n=== Per-query inspection (after W) ===")
    inspection = evaluator.inspect_predictions(
        df_test, h_o_hat_W, unique_h_o, unique_names, true_indices, top_k=5
    )
    print(inspection[["subject", "relation", "true_object", "true_rank", "top_1", "top_2", "top_3"]].to_string())

    # ── Save to Excel ────────────────────────────────────────────────────────
    base_path = config["results_paths"]["evaluation_results"]
    stem, ext = os.path.splitext(base_path)
    output_path = f"{stem}_{datetime.now().strftime('%Y%m%d')}{ext}"
    evaluator.save_results(metrics_df, inspection, output_path)
