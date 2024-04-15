'''Module to train the various transformations'''

import os
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import pandas as pd
import numpy as np

from dataload_batches import DataPreparation
from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS
from transformations import LinearTransformation, OrthogonalLayer, MultilayerPerceptron, RBFKernelLayer
from utils import UtilityFunctions
from config import get_config


# Set seeds for reproducibility
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def load_or_generate_embeddings(
        data_path: str,
        embeddings_path: str,
        model_name: str,
        utility_funcs: UtilityFunctions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(embeddings_path):
        return utility_funcs.load_embeddings_and_scores_from_torch(embeddings_path)
    else:
        print("Generating embeddings using CreateBERTEmbeddingsWithCLS")
        bert_embeddings = BERTEmbeddingsWithCLS(model_name)
        data_prep = DataPreparation(model_name, bert_embeddings)
        return data_prep.prepare_data(pd.read_csv(data_path))

def initialize_model(embedding_dim: int) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    #model = LinearTransformation(embedding_dim, embedding_dim) # choose the model here
    model = OrthogonalLayer(embedding_dim)
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    return model, loss_function, optimizer, scheduler

def compute_predicted_scores(transformed_embeddings: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        transformed_embeddings[::2], # select every other embedding starting from the first one
        transformed_embeddings[1::2], # select every other embedding starting from the second one
        dim=1                         # if we have e1, e2, e3, e4 it will compute the similarity between (e1, e2), (e3, e4)
    ).view(-1, 1)

def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_embeddings: torch.Tensor,
        train_scores: torch.Tensor,
        loss_function: nn.Module,
        utility_funcs: UtilityFunctions
        #lambda_orth: float = 0.1
    ) -> Tuple[torch.Tensor, float]:
    optimizer.zero_grad()
    transformed_embeddings = model(train_embeddings)
    print(transformed_embeddings.size())
    predicted_scores = compute_predicted_scores(transformed_embeddings)
    loss = loss_function(predicted_scores, train_scores)
    # orth_loss = utility_funcs.orthogonal_regularization(model, lambda_orth)
    # total_loss = loss + orth_loss
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss, utility_funcs.compute_pearson_correlation(predicted_scores, train_scores)

def validate(
        model: nn.Module,
        val_data: torch.Tensor,
        val_scores: torch.Tensor,
        loss_function: nn.Module,
        utility_funcs: UtilityFunctions
    ) -> Tuple[torch.Tensor, float]:
    with torch.no_grad():
        transformed_embeddings = model(val_data)
        predicted_scores = compute_predicted_scores(transformed_embeddings)
        loss = loss_function(predicted_scores, val_scores)
        return loss, utility_funcs.compute_pearson_correlation(predicted_scores, val_scores)

def main() -> None:
    config = get_config()
    train_data_path = config['data_paths']['train_data']
    val_data_path = config['data_paths']['val_data']
    model_name = config['model_name']
    embeddings_train_path = config['data_paths']['train_embeddings']
    embeddings_val_path = config['data_paths']['val_embeddings']
    utils = UtilityFunctions()
    
    train_embeddings, train_scores = load_or_generate_embeddings(train_data_path, embeddings_train_path, model_name, utils)
    val_embeddings, val_scores = load_or_generate_embeddings(val_data_path, embeddings_val_path, model_name, utils)
    
    model, loss_function, optimizer, scheduler = initialize_model(train_embeddings.shape[1])
    
    train_losses = []
    val_losses = []
    train_corrs = []
    val_corrs = []
    
    NUM_EPOCHS = config['num_epochs']
    early_stopping_patience = config['early_stopping_patience']
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_corr = train_epoch(model, optimizer, train_embeddings, train_scores, loss_function, utils)
        val_loss, val_corr = validate(model, val_embeddings, val_scores, loss_function, utils)
        
        scheduler.step(val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_corrs.append(train_corr)
        val_corrs.append(val_corr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {'model_class': OrthogonalLayer,
                 'state_dict': model.state_dict()
                }, config['model_paths']['saved_model_full_dim'])
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {train_loss.item()}, Pearson Correlation (Train): {train_corr}, Learning Rate: {current_lr}')
        print(f'Validation - Loss: {val_loss.item()}, Pearson Correlation (Validation): {val_corr} \n')
    
    # Plotting after the training loop
    utils.plot_losses(train_losses, val_losses)
    utils.plot_pearson_correlations(train_corrs, val_corrs)

    learned_transformation = list(model.parameters())[0].detach().numpy()
    print(learned_transformation.shape)
    print(learned_transformation)
    #print(utils.is_orthogonal(learned_transformation))


if __name__ == '__main__':
    main()
