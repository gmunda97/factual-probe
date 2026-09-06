"""
Train a linear transformation W such that W(h_s + h_r) ≈ h_o.

Loss: mean cosine distance = mean(1 - cos(W(h_s + h_r), h_o))

Usage:
    python train.py
"""

import os
import sys
import platform
from datetime import datetime

# ── Shared src must be on the path before any local imports ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS, ModernBERTEmbeddings, ModernBERTEmbeddingsWithCLS
from transformations import LinearTransformation
from utils import UtilityFunctions
from config import MODEL_SPEC, get_config


def cosine_loss(o_hat: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance between predicted and true object embeddings."""
    return (1 - F.cosine_similarity(o_hat, o)).mean()


def infonce_loss(o_hat: torch.Tensor, o: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute the InfoNCE loss for a batch of predicted and true object embeddings.

    Args:
        o_hat: Predicted object embeddings (N, D)
        o: True object embeddings (N, D)
        temperature: Scaling factor for logits

    Returns:
        InfoNCE loss (scalar)
    """
    q = F.normalize(o_hat, dim=1)       # (N, D)
    k = F.normalize(o, dim=1)           # (N, D)
    logits = (q @ k.T) / temperature    # (N, N) similarity matrix
    labels = torch.arange(len(q), device=q.device)
    return F.cross_entropy(logits, labels)


def infonce_loss_chunked(
    o_hat: torch.Tensor,
    o: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute InfoNCE against all targets without an N x N logits matrix."""
    k = F.normalize(o, dim=1)
    losses = []
    for start in range(0, len(o_hat), chunk_size):
        stop = start + chunk_size
        q_chunk = F.normalize(o_hat[start:stop], dim=1)
        logits = (q_chunk @ k.T) / temperature
        labels = torch.arange(start, min(stop, len(o_hat)), device=o_hat.device)
        losses.append(F.cross_entropy(logits, labels, reduction="sum"))
    return torch.stack(losses).sum() / len(o_hat)


def total_loss(
    o_hat: torch.Tensor,
    o: torch.Tensor,
    use_infonce: bool,
    infonce_lambda: float,
    infonce_temperature: float,
) -> torch.Tensor:
    loss = cosine_loss(o_hat, o)
    if use_infonce:
        loss = loss + infonce_lambda * infonce_loss(o_hat, o, infonce_temperature)
    return loss




# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_strings(
    encoder: BERTEmbeddings,
    texts: list[str],
    transformer_layer: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode *texts* from one transformer layer and mean-pool the tokens."""
    encoder.model.eval()
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[i : i + batch_size]
        inputs = encoder.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        with torch.no_grad():
            out = encoder.model(**inputs, output_hidden_states=True)
        layer_hidden_state = out.hidden_states[transformer_layer]
        all_vecs.append(encoder.pool(layer_hidden_state, inputs['attention_mask']))
    return torch.cat(all_vecs, dim=0)


def load_or_compute(
    encoder: BERTEmbeddings,
    texts: list[str],
    cache_path: str,
    transformer_layer: int,
) -> torch.Tensor:
    """Return cached embeddings if available, otherwise compute and save them."""
    if os.path.exists(cache_path):
        print(f"  [cache hit]  {os.path.basename(cache_path)}")
        return torch.load(cache_path, map_location="cpu")
    print(f"  [computing]  {os.path.basename(cache_path)}")
    vecs = embed_strings(encoder, texts, transformer_layer)
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
    start_time = datetime.now().astimezone()
    run_id = start_time.strftime("%Y%m%dT%H%M%S%z")
    metadata_path = os.path.join(
        config['results_paths']['run_metadata_dir'],
        f"{MODEL_SPEC}_{run_id}.json",
    )
    run_metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": start_time.isoformat(),
        "script": os.path.abspath(__file__),
        "git_commit": UtilityFunctions.get_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "config": config,
    }

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
    layer = config['transformer_layer']
    train_h_s = load_or_compute(encoder, train_df["subject"].tolist(),    config['cache_paths']['train_h_s'], layer)
    train_h_r = load_or_compute(encoder, train_df["pred_value"].tolist(), config['cache_paths']['train_h_r'], layer)
    train_h_o = load_or_compute(encoder, train_df["object"].tolist(),     config['cache_paths']['train_h_o'], layer)

    print("\n=== Val embeddings ===")
    val_h_s = load_or_compute(encoder, val_df["subject"].tolist(),    config['cache_paths']['val_h_s'], layer)
    val_h_r = load_or_compute(encoder, val_df["pred_value"].tolist(), config['cache_paths']['val_h_r'], layer)
    val_h_o = load_or_compute(encoder, val_df["object"].tolist(),     config['cache_paths']['val_h_o'], layer)

    train_h_o_hat = train_h_s + train_h_r   # (N_train, D)
    val_h_o_hat   = val_h_s   + val_h_r     # (N_val,   D)
    print(f"\nEmbedding dim : {train_h_o.shape[1]}")
    print(f"Train tensors : {train_h_o_hat.shape}")
    print(f"Val tensors   : {val_h_o_hat.shape}")

    # --- Model ---
    D = train_h_o_hat.shape[1]
    model     = LinearTransformation(D, D)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
    )

    train_loader = DataLoader(
        TensorDataset(train_h_o_hat, train_h_o),
        batch_size=config['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_h_o_hat, val_h_o),
        batch_size=config['batch_size'],
        shuffle=False,
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    # --- Training loop ---
    print("\n=== Training ===")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_queries, batch_targets in train_loader:
            optimizer.zero_grad()
            train_output = model(batch_queries)
            batch_loss = total_loss(
                train_output,
                batch_targets,
                config['use_infonce'],
                config['infonce_lambda'],
                config['infonce_temperature'],
            )
            batch_loss.backward()
            optimizer.step()
            train_loss_sum += batch_loss.item() * len(batch_queries)
            train_count += len(batch_queries)
        train_loss = train_loss_sum / train_count

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for batch_queries, batch_targets in val_loader:
                val_output = model(batch_queries)
                batch_loss = total_loss(
                    val_output,
                    batch_targets,
                    config['use_infonce'],
                    config['infonce_lambda'],
                    config['infonce_temperature'],
                )
                val_loss_sum += batch_loss.item() * len(batch_queries)
                val_count += len(batch_queries)
            val_loss = val_loss_sum / val_count

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
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
                  f"train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  "
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

    metrics_before = evaluate(val_h_o_hat, val_h_o)
    metrics_after  = evaluate(val_pred_W,  val_h_o)

    metrics_df = pd.DataFrame(
        {"Before W": metrics_before, "After W": metrics_after}
    ).T.round(4)
    print(metrics_df)

    run_metadata.update({
        "status": "completed",
        "finished_at": datetime.now().astimezone().isoformat(),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "embedding_dimension": D,
        "epochs_completed": len(train_losses),
        "best_validation_loss": best_val_loss,
        "metrics": metrics_df.to_dict(),
    })
    UtilityFunctions.save_run_metadata(metadata_path, run_metadata)
    print(f"Completed run metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()