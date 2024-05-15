'''Python file for utility functions'''

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


class UtilityFunctions:

    @staticmethod
    def compute_pearson_correlation(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        predicted = predicted.squeeze().cpu().detach().numpy()
        actual = actual.squeeze().cpu().detach().numpy()
        corr, _ = pearsonr(predicted, actual)
        return corr

    @staticmethod
    def compute_spearman_correlation(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        predicted = predicted.squeeze().cpu().detach().numpy()
        actual = actual.squeeze().cpu().detach().numpy()
        corr, _ = spearmanr(predicted, actual)
        return corr
    
    @staticmethod
    def compute_mean_squared_error(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        predicted = predicted.squeeze().cpu().detach().numpy()
        actual = actual.squeeze().cpu().detach().numpy()
        mse = mean_squared_error(actual, predicted)
        return mse
    
    @staticmethod
    def compute_root_mean_squared_error(predicted: torch.Tensor, actual: torch.Tensor) -> float:
        predicted = predicted.squeeze().cpu().detach().numpy()
        actual = actual.squeeze().cpu().detach().numpy()
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def is_orthogonal(matrix: np.ndarray) -> bool:
        return np.allclose(matrix.T @ matrix, np.eye(matrix.shape[1]), atol=1e-5)
    
    @staticmethod
    def orthogonal_regularization(model: nn.Module, lambda_orth: float = 0.0001) -> torch.Tensor:
        orth_loss = 0.0
        for param in model.parameters():
            if param.requires_grad and param.data.shape[0] == param.data.shape[1]:  # Check for square matrices
                identity_matrix = torch.eye(param.data.shape[0], requires_grad=False)
                orth_loss += (torch.norm(torch.mm(param, torch.transpose(param, 0, 1)) - identity_matrix, p='fro') ** 2)
        
        return lambda_orth * orth_loss

    @staticmethod
    def load_embeddings_and_scores_from_torch(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        loaded_data = torch.load(file_path)
        embeddings = loaded_data['embeddings']
        similarity_scores = loaded_data['similarity_scores']
        return embeddings, similarity_scores
    
    @staticmethod
    def get_batch(data: torch.Tensor, batch_idx: int, effective_batch_size: int) -> torch.Tensor:
        start_idx = batch_idx * effective_batch_size
        end_idx = start_idx + effective_batch_size
        return data[start_idx:end_idx]
    
    @staticmethod
    def calculate_n_batches(total_size: int, batch_size: int) -> int:
        return (total_size + batch_size -1) // batch_size

    @staticmethod
    def plot_losses(train_losses: list, val_losses: list) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig('./../resources/plots/wiki_desc/full_dim/loss.png')

    @staticmethod
    def plot_pearson_correlations(train_corrs: list, val_corrs: list) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_corrs, label='Training Pearson Correlation')
        plt.plot(val_corrs, label='Validation Pearson Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Pearson Correlation over Epochs')
        plt.legend()
        plt.savefig('./../resources/plots/wiki_desc/full_dim/pearson_correlation.png')
    
    @staticmethod
    def plot_all_losses(all_train_losses: dict, all_val_losses: dict) -> None:
        plt.figure(figsize=(12, 6))
        for output_dim, losses in all_train_losses.items():
            plt.plot(losses, label=f'Training Loss (dim={output_dim})')
        for output_dim, losses in all_val_losses.items():
            plt.plot(losses, label=f'Validation Loss (dim={output_dim})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs by Output Dimension')
        plt.legend()
        plt.savefig('./../resources/plots/wiki_desc/reduced_dim/all_losses.png')
    
    @staticmethod
    def plot_all_pearson_correlations(all_train_corrs: dict, all_val_corrs: dict) -> None:
        plt.figure(figsize=(12, 6))
        for output_dim, corrs in all_train_corrs.items():
            plt.plot(corrs, label=f'Training Pearson Correlation (dim={output_dim})')
        for output_dim, corrs in all_val_corrs.items():
            plt.plot(corrs, label=f'Validation Pearson Correlation (dim={output_dim})')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Pearson Correlation over Epochs by Output Dimension')
        plt.legend()
        plt.savefig('./../resources/plots/wiki_desc/reduced_dim/all_pearson_correlations.png')
