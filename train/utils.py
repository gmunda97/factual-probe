"""Python file for utility functions"""

from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


class UtiliyFunctions:

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
    def is_orthogonal(matrix: np.ndarray) -> bool:
        return np.allclose(matrix.T @ matrix, np.eye(matrix.shape[1]), atol=1e-5)

    @staticmethod
    def load_embeddings_and_scores_from_torch(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        loaded_data = torch.load(file_path)
        embeddings = loaded_data['embeddings']
        similarity_scores = loaded_data['similarity_scores']

        return embeddings, similarity_scores

    @staticmethod
    def plot_losses(train_losses: list, val_losses: list) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig('./resources/plots/loss.png')

    @staticmethod
    def plot_pearson_correlations(train_corrs: list, val_corrs: list) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_corrs, label='Training Pearson Correlation')
        plt.plot(val_corrs, label='Validation Pearson Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Pearson Correlation over Epochs')
        plt.legend()
        plt.savefig('./resources/plots/pearson_correlation.png')
