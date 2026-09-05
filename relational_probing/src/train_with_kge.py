"""
Train a linear transformation W such that W(h_s + h_r) ≈ h_o, but h_o is computed using a knowledge graph embedding (KGE) model instead of an LLM.

Loss: mean cosine distance = mean(1 - cos(W(h_s + h_r), h_o))

Usage:
    python train.py
"""

import os
import sys

# ── Shared src must be on the path before any local imports ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS, ModernBERTEmbeddings, ModernBERTEmbeddingsWithCLS
from transformations import LinearTransformation
from utils import UtilityFunctions
from config import get_config


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_strings(encoder: BERTEmbeddingsWithCLS, texts: list[str], batch_size: int = 64) -> torch.Tensor:
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


def parse_kge_embeddings(df: pd.DataFrame) -> torch.Tensor:
    """Convert the space-separated obj_embedding strings to a (N, D_kge) tensor."""
    return torch.tensor(
        df['obj_embedding'].apply(lambda s: list(map(float, s.split()))).tolist(),
        dtype=torch.float32,
    )


def load_or_compute(encoder: BERTEmbeddingsWithCLS, texts: list[str], cache_path: str) -> torch.Tensor:
    """Return cached embeddings if available, otherwise compute and save them."""
    if os.path.exists(cache_path):
        print(f"  [cache hit]  {os.path.basename(cache_path)}")
        return torch.load(cache_path, map_location="cpu")
    print(f"  [computing]  {os.path.basename(cache_path)}")
    vecs = embed_strings(encoder, texts)
    torch.save(vecs, cache_path)
    return vecs


def evaluate(queries: torch.Tensor, targets: torch.Tensor, k_list: tuple = (1, 3, 10)) -> dict:
    """
    Retrieval metrics for the triples-completion task.

    For each query i the true candidate is targets[i]; ranking is done against
    the full targets pool (N × N similarity matrix).

    Returns: mean_cos_sim, MRR, Hits@k for each k in k_list.
    """
    with torch.no_grad():
        q_norm = F.normalize(queries, dim=1)
        t_norm = F.normalize(targets, dim=1)

        mean_cos    = F.cosine_similarity(q_norm, t_norm).mean().item()
        sim_matrix  = q_norm @ t_norm.T                             # (N, N)
        true_scores = sim_matrix.diagonal().unsqueeze(1)            # (N, 1)
        ranks       = (sim_matrix > true_scores).sum(dim=1) + 1     # (N,) 1-indexed

        mrr    = (1.0 / ranks.float()).mean().item()
        hits_k = {k: (ranks <= k).float().mean().item() for k in k_list}

    return {"mean_cos_sim": mean_cos, "MRR": mrr, **{f"Hits@{k}": v for k, v in hits_k.items()}}


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    config = get_config()

    # --- Data ---
    train_df = pd.read_csv(config['data_paths']['train'])[config['cols']]
    val_df   = pd.read_csv(config['data_paths']['val'])[config['cols']]
    print(f"Train : {len(train_df):,} triples")
    print(f"Val   : {len(val_df):,} triples")

    # --- Encoder ---
    encoder = BERTEmbeddings(config['model_name'])
    encoder.model.eval()
    print(f"Loaded {config['model_name']} — hidden size: {encoder.model.config.hidden_size}")

    # --- Embeddings (cached) ---
    print("\n=== Train embeddings ===")
    train_h_s = load_or_compute(encoder, train_df["subject"].tolist(),    config['cache_paths']['train_h_s'])
    train_h_r = load_or_compute(encoder, train_df["pred_value"].tolist(), config['cache_paths']['train_h_r'])
    train_h_o = parse_kge_embeddings(train_df)

    print("\n=== Val embeddings ===")
    val_h_s = load_or_compute(encoder, val_df["subject"].tolist(),    config['cache_paths']['val_h_s'])
    val_h_r = load_or_compute(encoder, val_df["pred_value"].tolist(), config['cache_paths']['val_h_r'])
    val_h_o = parse_kge_embeddings(val_df)

    train_h_o_hat = train_h_s + train_h_r   # (N_train, D)
    val_h_o_hat   = val_h_s   + val_h_r     # (N_val,   D)
    print(f"\nInput dim  (h_s + h_r) : {train_h_o_hat.shape[1]}")
    print(f"Target dim (KGE h_o)   : {train_h_o.shape[1]}")
    print(f"Train tensors : {train_h_o_hat.shape}")
    print(f"Val tensors   : {val_h_o_hat.shape}")

    # --- Model ---
    D_in  = train_h_o_hat.shape[1]   # LM hidden size (e.g. 768)
    D_out = train_h_o.shape[1]        # KGE dim (e.g. 100)
    model     = LinearTransformation(D_in, D_out)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    # --- Training loop ---
    print("\n=== Training ===")
    for epoch in range(config['num_epochs']):
        model.train()
        optimizer.zero_grad()
        train_loss = (1 - F.cosine_similarity(model(train_h_o_hat), train_h_o)).mean()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = (1 - F.cosine_similarity(model(val_h_o_hat), val_h_o)).mean()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss    = val_loss.item()
            patience_counter = 0
            torch.save({"state_dict": model.state_dict()},
                       config['model_paths']['saved_model'])
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>3}/{config['num_epochs']}  "
                  f"train_loss={train_loss.item():.6f}  "
                  f"val_loss={val_loss.item():.6f}  "
                  f"lr={current_lr:.2e}")

    print(f"\nBest val loss: {best_val_loss:.6f}  (saved → {config['model_paths']['saved_model']})")

    # --- Loss curves ---
    utils = UtilityFunctions()
    utils.plot_losses_relational_probing(train_losses, val_losses, config['model_paths']['loss_curve'])

    # --- Evaluation ---
    print("\n=== Evaluation on validation set ===")
    best_ckpt = torch.load(config['model_paths']['saved_model'], map_location="cpu", weights_only=True)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        val_pred_W = model(val_h_o_hat)

    metrics = evaluate(val_pred_W, val_h_o)

    metrics_df = pd.DataFrame({"After W": metrics}).T.round(4)
    print(metrics_df)


if __name__ == "__main__":
    main()