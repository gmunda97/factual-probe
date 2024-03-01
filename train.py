"""Module to train the linear transformation"""

import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from dataload_batches import DataPreparation
from bert_embeddings import CreateBERTEmbeddings, CreateBERTEmbeddingsWithCLS
from linear_transformations import LinearTransformation, OrthogonalLayer

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


if __name__ == '__main__':

    train_data = pd.read_csv('./data/dataset/wikidata5m_6k_train.csv')
    val_data = pd.read_csv('data/dataset/wikidata5m_6k_val.csv')

    MODEL_NAME = 'bert-base-uncased'
    EMBEDDINGS_PICKLE_TRAIN = 'data/embeddings/embeddings_train2.pkl'
    EMBEDDINGS_PICKLE_VAL = 'data/embeddings/embeddings_val2.pkl'

    if os.path.exists(EMBEDDINGS_PICKLE_TRAIN):
        with open(EMBEDDINGS_PICKLE_TRAIN, 'rb') as file:
            saved_train_embeddings = pickle.load(file)
        with open(EMBEDDINGS_PICKLE_VAL, 'rb') as file:
            saved_val_embeddings = pickle.load(file)
        
        print("Using saved embeddings\n")
        normalized_embeddings_train = saved_train_embeddings
        normalized_embeddings_val = saved_val_embeddings
        similarity_scores_train = torch.Tensor(train_data['similarity'].values).view(-1, 1)
        similarity_scores_val = torch.Tensor(val_data['similarity'].values).view(-1, 1)
    else:
        print("Generating embeddings using CreateBERTEmbeddingsWithCLS\n")
        bert_embeddings = CreateBERTEmbeddingsWithCLS(MODEL_NAME)
        data_prep = DataPreparation(MODEL_NAME, bert_embeddings)
        normalized_embeddings_train, similarity_scores_train = data_prep.prepare_data(train_data[:100])
        normalized_embeddings_val, similarity_scores_val = data_prep.prepare_data(val_data[:20])

    EMBEDDING_DIM = normalized_embeddings_train.shape[1]

    linear_transformation = LinearTransformation(EMBEDDING_DIM, EMBEDDING_DIM)
    #linear_transformation = OrthogonalLayer(EMBEDDING_DIM)

    loss_function = nn.MSELoss()
    #optimizer = optim.SGD(linear_transformation.parameters(), lr=5.0, momentum=0.05) #weight_decay=0.005)
    optimizer = optim.AdamW(linear_transformation.parameters(), lr=0.05, weight_decay=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    NUM_EPOCHS = 100
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    writer = SummaryWriter('./runs/linear_transformation_experiment')

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        transformed_embeddings = linear_transformation(normalized_embeddings_train)
        print(f"Shape of the transformed embeddings: {transformed_embeddings.shape}")
        
        predicted_scores_train = F.cosine_similarity(
            transformed_embeddings[::2], # select every other embedding starting from the first one
            transformed_embeddings[1::2], # select every other embedding starting from the second one
            dim=1                         # if we have e1, e2, e3, e4 it will compute the similarity between (e1, e2), (e3, e4)
        ).view(-1, 1)

        print(f"Shape of the predicted scores: {predicted_scores_train.shape}")

        loss = loss_function(predicted_scores_train, similarity_scores_train)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Calculate the Pearson correlation coefficient for evaluation
        corr_train = compute_pearson_correlation(predicted_scores_train, similarity_scores_train)

        # Evaluation on the validation set
        with torch.no_grad():
            transformed_embeddings_val = linear_transformation(normalized_embeddings_val)
            predicted_scores_val = F.cosine_similarity(
                transformed_embeddings_val[::2],
                transformed_embeddings_val[1::2],
                dim=1
            ).view(-1, 1)

            loss_val = loss_function(predicted_scores_val, similarity_scores_val)
            corr_val = compute_pearson_correlation(predicted_scores_val, similarity_scores_val)

            scheduler.step(loss_val)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epoch)

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                torch.save(
                    {'model_class': LinearTransformation,
                     'state_dict': linear_transformation.state_dict()
                    }, 'trained_models/best_linear_transformation.pth')
            else:
                patience_counter += 1
                if patience_counter > early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}, Pearson Correlation (Train): {corr_train}, Learning Rate: {current_lr}')
        print(f'Validation - Loss: {loss_val.item()}, Pearson Correlation (Validation): {corr_val} \n')


    learned_transformation = list(linear_transformation.parameters())[0].detach().numpy()
    print(learned_transformation.shape)
