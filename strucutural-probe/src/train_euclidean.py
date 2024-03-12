"""Module to train the linear transformation"""

import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from dataload_batches import DataPreparation
from bert_embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS
from transformations import LinearTransformation, OrthogonalLayer, MultilayerPerceptron, RBFKernelLayer

# Set seeds for reproducibility
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def compute_pearson_correlation(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    predicted = predicted.squeeze().cpu().detach().numpy()
    actual = actual.squeeze().cpu().detach().numpy()
    corr, _ = pearsonr(predicted, actual)

    return corr

def compute_spearman_correlation(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    predicted = predicted.squeeze().cpu().detach().numpy()
    actual = actual.squeeze().cpu().detach().numpy()
    corr, _ = spearmanr(predicted, actual)

    return corr

def load_embeddings_and_scores_from_torch(file_path: str):
    loaded_data = torch.load(file_path)
    embeddings = loaded_data['embeddings']
    similarity_scores = loaded_data['similarity_scores']

    return embeddings, similarity_scores


if __name__ == '__main__':

    train_data = pd.read_csv('./../data/dataset/wikidata5m_42k_train.csv')
    val_data = pd.read_csv('./../data/dataset/wikidata5m_42k_valid.csv')

    MODEL_NAME = 'bert-base-uncased'
    EMBEDDINGS_PYTORCH_TRAIN = './../data/embeddings/wikidata5m_42k_train_embeddings.pt'
    EMBEDDINGS_PYTORCH_VAL = './../data/embeddings/wikidata5m_42k_valid_embeddings.pt'

    if os.path.exists(EMBEDDINGS_PYTORCH_TRAIN):
        saved_train_embeddings, similarity_scores_train = load_embeddings_and_scores_from_torch(EMBEDDINGS_PYTORCH_TRAIN)
        saved_val_embeddings, similarity_scores_val = load_embeddings_and_scores_from_torch(EMBEDDINGS_PYTORCH_VAL)
        
        print("Using saved embeddings\n")
        normalized_embeddings_train = saved_train_embeddings
        print(normalized_embeddings_train.shape)
        print(type(normalized_embeddings_train))
        normalized_embeddings_val = saved_val_embeddings
        print(normalized_embeddings_val.shape)
        print(type(normalized_embeddings_val))
        # similarity_scores_train = torch.Tensor(train_data['similarity'].values).view(-1, 1)
        # similarity_scores_val = torch.Tensor(val_data['similarity'].values).view(-1, 1)
    else:
        print("Generating embeddings using CreateBERTEmbeddingsWithCLS\n")
        bert_embeddings = BERTEmbeddingsWithCLS(MODEL_NAME)
        data_prep = DataPreparation(MODEL_NAME, bert_embeddings)
        normalized_embeddings_train, similarity_scores_train = data_prep.prepare_data(train_data)
        normalized_embeddings_val, similarity_scores_val = data_prep.prepare_data(val_data)

    print(f"Shape of the normalized embeddings: {normalized_embeddings_train.shape}")
    EMBEDDING_DIM = normalized_embeddings_train.shape[1]
    HIDDEN_DIM = 512
    RBF_FEAUTURES = 100

    model = LinearTransformation(EMBEDDING_DIM, EMBEDDING_DIM)
    #model = RBFKernelLayer(EMBEDDING_DIM, RBF_FEAUTURES, EMBEDDING_DIM)
    #model = MultilayerPerceptron(EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DIM)
    #model = OrthogonalLayer(EMBEDDING_DIM)

    loss_function = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=5.0, momentum=0.05) #weight_decay=0.005)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    NUM_EPOCHS = 100
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        transformed_embeddings = model(normalized_embeddings_train)
        print(f"Shape of the transformed embeddings: {transformed_embeddings.shape}")
        
        predicted_scores_train = torch.norm(
            transformed_embeddings[::2] - transformed_embeddings[1::2],
            dim=1,
            p=2
        ).view(-1, 1)

        print(f"Shape of the predicted scores: {predicted_scores_train.shape}")

        loss = loss_function(predicted_scores_train, similarity_scores_train)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Calculate the Pearson correlation coefficient for evaluation
        corr_train = compute_pearson_correlation(predicted_scores_train, similarity_scores_train)

        # Evaluation on the validation set
        with torch.no_grad():
            transformed_embeddings_val = model(normalized_embeddings_val)
            predicted_scores_val = torch.norm(
                transformed_embeddings_val[::2] - transformed_embeddings_val[1::2],
                dim=1, 
                p=2
            ).view(-1, 1)

            loss_val = loss_function(predicted_scores_val, similarity_scores_val)
            corr_val = compute_pearson_correlation(predicted_scores_val, similarity_scores_val)

            scheduler.step(loss_val)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                torch.save(
                    {'model_class': LinearTransformation,
                     'state_dict': model.state_dict()
                    }, './../trained_models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter > early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}, Pearson Correlation (Train): {corr_train}, Learning Rate: {current_lr}')
        print(f'Validation - Loss: {loss_val.item()}, Pearson Correlation (Validation): {corr_val} \n')


    learned_transformation = list(model.parameters())[0].detach().numpy()
    print(learned_transformation.shape)
