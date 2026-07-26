"""
Evaluate the trained relational-probing linear transformation on the test set.

Metrics reported (before and after applying W):
  - Mean cosine similarity  cos(query, h_o)
  - Mean Reciprocal Rank    (MRR)
  - Hits@k                  for k in {1, 3, 10}

Usage:
    python evaluate.py
"""

import os
import sys
from typing import Optional

# ── Shared src must be on the path before any local imports ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS, ModernBERTEmbeddings, ModernBERTEmbeddingsWithCLS
from transformations import LinearTransformation

COLS = ["subject", "pred_value", "object"]


class Evaluator:
    def __init__(
        self,
        data_path: str,
        cache_dir: str,
        model_path: str,
        model_name: str = "answerdotai/ModernBERT-base",
        cache_prefix: str = "42k_test_rel",
        k_list: tuple = (1, 3, 10),
    ) -> None:
        self.data_path    = data_path
        self.cache_dir    = cache_dir
        self.model_path   = model_path
        self.model_name   = model_name
        self.cache_prefix = cache_prefix
        self.k_list       = k_list
        self._encoder: Optional[ModernBERTEmbeddings] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def encoder(self) -> ModernBERTEmbeddings:
        """Lazy-load the BERT encoder once."""
        if self._encoder is None:
            print(f"Loading encoder: {self.model_name}")
            self._encoder = ModernBERTEmbeddings(self.model_name)
            self._encoder.model.eval()
        return self._encoder

    def _embed_strings(self, encoder: ModernBERTEmbeddings, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Encode *texts* in batches. Returns a (N, D) mean-pooled tensor."""
        encoder.model.eval()
        all_vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
            batch = texts[i : i + batch_size]
            inputs = encoder.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            with torch.no_grad():
                out = encoder.model(**inputs)
            all_vecs.append(encoder.pool(out.last_hidden_state, inputs['attention_mask']))
        return torch.cat(all_vecs, dim=0)

    def _load_or_compute(self, texts: list[str], cache_path: str) -> torch.Tensor:
        """Return cached embeddings if available, otherwise compute and save."""
        if os.path.exists(cache_path):
            print(f"  [cache hit]  {os.path.basename(cache_path)}")
            return torch.load(cache_path, map_location="cpu")
        print(f"  [computing]  {os.path.basename(cache_path)}")
        vecs = self._embed_strings(self.encoder, texts)
        torch.save(vecs, cache_path)
        return vecs

    # ── Public API ────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)[COLS]
        print(f"Test set: {len(df):,} triples  ({self.data_path})")
        return df

    def load_embeddings(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (h_s, h_r, h_o) tensors for every row in *df*.
        Each tensor has shape (N, D). Results are cached to disk.
        """
        h_s = self._load_or_compute(
            df["subject"].tolist(),
            os.path.join(self.cache_dir, f"{self.cache_prefix}_h_s_modernbert.pt"),
        )
        h_r = self._load_or_compute(
            df["pred_value"].tolist(),
            os.path.join(self.cache_dir, f"{self.cache_prefix}_h_r_modernbert.pt"),
        )
        h_o = self._load_or_compute(
            df["object"].tolist(),
            os.path.join(self.cache_dir, f"{self.cache_prefix}_h_o_modernbert.pt"),
        )
        return h_s, h_r, h_o

    def load_model(self, embedding_dim: int) -> LinearTransformation:
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model = LinearTransformation(embedding_dim, embedding_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Loaded model from {self.model_path}")
        return model

    def build_unique_object_pool(
        self, df: pd.DataFrame, h_o: torch.Tensor
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Deduplicate the object pool so that each unique entity name is
        represented by exactly one vector (its first occurrence in the dataset).

        This avoids the problem where the same entity string appears as the
        object of many triples: without deduplication, identical vectors would
        crowd the top ranks and distort MRR / Hits@k.

        Returns:
            unique_names  : list of U unique object strings
            unique_h_o    : (U, D) tensor — one embedding per unique object
            true_indices  : (N,) LongTensor — for triple i, the index of
                            its true object inside unique_h_o
        """
        object_names = df["object"].tolist()

        # Map each name to the index of its first occurrence in the full list
        first_occurrence: dict[str, int] = {}
        for idx, name in enumerate(object_names):
            if name not in first_occurrence:
                first_occurrence[name] = idx

        unique_names   = list(first_occurrence.keys())          # length U
        unique_row_ids = list(first_occurrence.values())        # row indices in h_o
        unique_h_o     = h_o[unique_row_ids]                    # (U, D)

        name_to_unique_idx = {name: i for i, name in enumerate(unique_names)}
        true_indices = torch.tensor(
            [name_to_unique_idx[name] for name in object_names], dtype=torch.long
        )  # (N,)

        print(f"  Object pool: {len(object_names):,} triples → {len(unique_names):,} unique objects")
        return unique_names, unique_h_o, true_indices

    def compute_metrics(
        self,
        queries: torch.Tensor,         # (N, D) — one per triple
        unique_targets: torch.Tensor,  # (U, D) — one per unique object
        true_indices: torch.Tensor,    # (N,)   — index of the true object in unique_targets
    ) -> dict[str, float]:
        """
        Retrieval metrics against the deduplicated object pool.

        For each query i, the true object is unique_targets[true_indices[i]].
        Ranking is performed against all U unique objects.
        """
        with torch.no_grad():
            q_norm = F.normalize(queries, dim=1)         # (N, D)
            t_norm = F.normalize(unique_targets, dim=1)  # (U, D)

            # Cosine sim between each query and its own true object
            true_vecs = unique_targets[true_indices]     # (N, D)
            mean_cos  = F.cosine_similarity(
                F.normalize(queries, dim=1),
                F.normalize(true_vecs, dim=1)
            ).mean().item()

            sim_matrix  = q_norm @ t_norm.T              # (N, U)
            true_scores = sim_matrix[
                torch.arange(len(queries)), true_indices
            ].unsqueeze(1)                               # (N, 1)
            ranks = (sim_matrix > true_scores).sum(dim=1) + 1  # (N,) 1-indexed

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
        """
        For each selected query, show the top-k retrieved object names from the
        deduplicated pool and highlight the true answer.

        Args:
            df             : the DataFrame ("subject", "pred_value", "object")
            queries        : (N, D) predicted vectors (ĥ or Wĥ)
            unique_targets : (U, D) deduplicated object embeddings
            unique_names   : list of U unique object strings
            true_indices   : (N,) index of the true object in unique_targets
            query_indices  : which queries to inspect; defaults to the first 5
            top_k          : how many candidates to show per query
        """
        if query_indices is None:
            query_indices = list(range(min(5, len(df))))

        rows = []
        with torch.no_grad():
            q_norm = F.normalize(queries, dim=1)         # (N, D)
            t_norm = F.normalize(unique_targets, dim=1)  # (U, D)
            sim_matrix = q_norm @ t_norm.T               # (N, U)

        for i in query_indices:
            sims          = sim_matrix[i]                             # (U,)
            true_uid      = true_indices[i].item()                   # index in unique pool
            true_sim      = sims[true_uid].item()

            sorted_uids   = sims.argsort(descending=True).tolist()   # sorted unique-pool indices
            true_rank     = sorted_uids.index(true_uid) + 1          # 1-indexed

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

    def run_evaluation(self) -> pd.DataFrame:
        """
        Full evaluation pipeline. Returns a DataFrame with before/after metrics.
        """
        df             = self.load_data()
        h_s, h_r, h_o = self.load_embeddings(df)
        h_o_hat        = h_s + h_r   # additive composition (N, D)

        unique_names, unique_h_o, true_indices = self.build_unique_object_pool(df, h_o)

        model = self.load_model(embedding_dim=h_o.shape[1])
        with torch.no_grad():
            h_o_hat_W = model(h_o_hat)   # W(h_s + h_r),  (N, D)

        metrics_before = self.compute_metrics(h_o_hat,   unique_h_o, true_indices)
        metrics_after  = self.compute_metrics(h_o_hat_W, unique_h_o, true_indices)

        results_df = pd.DataFrame(
            {"Before W": metrics_before, "After W": metrics_after}
        ).T.round(4)

        print("\n=== Evaluation results ===")
        print(results_df.to_string())
        return results_df


if __name__ == "__main__":
    evaluator = Evaluator(
        data_path=os.path.join(_ROOT, "data", "dataset", "wikidata5m_42k_test_relations.csv"),
        cache_dir=os.path.join(_ROOT, "data", "embeddings"),
        model_path=os.path.join(_ROOT, "trained_models", "relational_probing", "transform_triples_modernbert.pt"),
    )

    df_test        = evaluator.load_data()
    h_s, h_r, h_o  = evaluator.load_embeddings(df_test)
    h_o_hat        = h_s + h_r

    unique_names, unique_h_o, true_indices = evaluator.build_unique_object_pool(df_test, h_o)

    model = evaluator.load_model(embedding_dim=h_o.shape[1])
    with torch.no_grad():
        h_o_hat_W = model(h_o_hat)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    evaluator.run_evaluation()

    # ── Per-query inspection (first 5 queries, top-5 candidates) ─────────────
    print("\n=== Per-query inspection (before W) ===")
    inspection_before = evaluator.inspect_predictions(
        df_test, h_o_hat, unique_h_o, unique_names, true_indices, top_k=5
    )
    print(inspection_before[["subject", "relation", "true_object", "true_rank", "top_1", "top_2", "top_3"]].to_string())

    print("\n=== Per-query inspection (after W) ===")
    inspection_after = evaluator.inspect_predictions(
        df_test, h_o_hat_W, unique_h_o, unique_names, true_indices, top_k=5
    )
    print(inspection_after[["subject", "relation", "true_object", "true_rank", "top_1", "top_2", "top_3"]].to_string())
