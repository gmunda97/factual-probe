'''Module to evaluate the trained models on the test data'''

from typing import Tuple, Dict, Optional
import pandas as pd
import torch
import torch.nn.functional as F

from utils import UtilityFunctions


class Evaluator:
    def __init__(self, embeddings_path: str, test_data_path: str, trained_model_path: Optional[str] = None) -> None:
        self.embeddings_path = embeddings_path
        self.test_data_path = test_data_path
        self.trained_model_path = trained_model_path
        self.utils = UtilityFunctions()
    
    def load_test_data(self) -> pd.DataFrame:
        return pd.read_csv(self.test_data_path)
    
    def load_embeddings_and_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        saved_test_embeddings, similarity_scores_test = self.utils.load_embeddings_and_scores_from_torch(self.embeddings_path)
        normalized_embeddings_test = saved_test_embeddings
        return similarity_scores_test, normalized_embeddings_test
    
    def load_model(self, normalized_embeddings_test: torch.Tensor) -> Optional[torch.nn.Module]:
        if self.trained_model_path is None:
            return None
        checkpoint = torch.load(self.trained_model_path)
        embedding_dim = normalized_embeddings_test.shape[1]
        if 'orthogonal' in self.trained_model_path:
            loaded_transformation = checkpoint['model_class'](embedding_dim)
        else:
            loaded_transformation = checkpoint['model_class'](embedding_dim, 512)
        loaded_transformation.load_state_dict(checkpoint['state_dict'])
        loaded_transformation.eval()
        return loaded_transformation
    
    def evaluate(
            self,
            normalized_embeddings_test: torch.Tensor,
            similarity_scores_test: torch.Tensor,
            loaded_transformation: Optional[torch.nn.Module] = None
        ) -> Dict[str, float]:
        if loaded_transformation is not None:
            transformed_embeddings = loaded_transformation(normalized_embeddings_test)
            predicted_similarity_scores = F.cosine_similarity(
                transformed_embeddings[::2],
                transformed_embeddings[1::2],
                dim=1
            ).view(-1, 1)
        else:
            predicted_similarity_scores = F.cosine_similarity(
                normalized_embeddings_test[::2],
                normalized_embeddings_test[1::2],
                dim=1
            ).view(-1, 1)

        results = {
            "Person correlation": self.utils.compute_pearson_correlation(
                predicted_similarity_scores, similarity_scores_test),
            "Spearman correlation": self.utils.compute_spearman_correlation(
                predicted_similarity_scores, similarity_scores_test),
            "Mean squared error": self.utils.compute_mean_squared_error(
                predicted_similarity_scores, similarity_scores_test),
            "Root mean squared error": self.utils.compute_root_mean_squared_error(
                predicted_similarity_scores, similarity_scores_test)
        }
        return results
    
    def run_evaluation(self) -> None:
        test_data = self.load_test_data()
        similarity_scores_test, normalized_embeddings_test = self.load_embeddings_and_scores()
        loaded_transformation = self.load_model(normalized_embeddings_test)
        results = self.evaluate(normalized_embeddings_test, similarity_scores_test, loaded_transformation)
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":

    evaluator = Evaluator(
        embeddings_path='./../../data/embeddings/wikidata5m_42k_test_embeddings_bart.pt',
        test_data_path='./../../data/dataset/wikidata5m_42k_test.csv',
        trained_model_path='./../../trained_models/reduced_dim/bart/best_model_dim_512.pth'
    )
    evaluator.run_evaluation()
