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
            loaded_transformation = checkpoint['model_class'](embedding_dim, embedding_dim)
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
        
        pearson_corr = self.utils.compute_pearson_correlation(predicted_similarity_scores, similarity_scores_test)
        spearman_corr = self.utils.compute_spearman_correlation(predicted_similarity_scores, similarity_scores_test)
        mse = self.utils.compute_mean_squared_error(predicted_similarity_scores, similarity_scores_test)
        rmse = self.utils.compute_root_mean_squared_error(predicted_similarity_scores, similarity_scores_test)

        results_df = pd.DataFrame({
            'predicted': predicted_similarity_scores.detach().squeeze().cpu().numpy(),
            'actual': similarity_scores_test.detach().squeeze().cpu().numpy()
        })
        results_df['pearson_residual'] = results_df['predicted'] - results_df['actual']
        results_df['spearman_residual'] = results_df['predicted'].rank() - results_df['actual'].rank()
        
        lowest_pearson = results_df.sort_values(by='pearson_residual').head(20)
        lowest_spearman = results_df.sort_values(by='spearman_residual').head(20)
        results_df.to_csv('./../resources/predictions/predictions_bert_cls.csv', index=False)

        # Print the results with the lowest correlations
        print("Lowest Pearson Correlation Data Points:")
        print(lowest_pearson)
        print("\nLowest Spearman Correlation Data Points:")
        print(lowest_spearman)

        results = {
            "Person correlation": pearson_corr,
            "Spearman correlation": spearman_corr,
            "Mean squared error": mse,
            "Root mean squared error": rmse
        }
        return results
    
    def run_evaluation(self) -> None:
        test_data = self.load_test_data()
        similarity_scores_test, normalized_embeddings_test = self.load_embeddings_and_scores()
        loaded_transformation = self.load_model(normalized_embeddings_test)
        results = self.evaluate(normalized_embeddings_test, similarity_scores_test, loaded_transformation)
        for key, value in results.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":

    evaluator = Evaluator(
        embeddings_path='./../../data/embeddings/wikidata5m_42k_test_embeddings_bert_cls.pt',
        test_data_path='./../../data/dataset/wikidata5m_42k_test.csv',
        trained_model_path='./../../trained_models/entities_only/full_dim/42k_linear_bert_cls.pth'
    )
    evaluator.run_evaluation()
