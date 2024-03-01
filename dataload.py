"""
Module to process the dataset for the model training
"""

import torch
import pandas as pd
from bert_embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS



class DataPreparation:
    def __init__(self, model_name: str, contextual_embeddings: torch.Tensor) -> None:
        self.model_name = model_name
        self.contextual_embeddings = contextual_embeddings

    def get_normalized_embeddings(self, entity_text: str) -> torch.Tensor:
        '''
        Method to L2 normalize embeddings for the given entity text
        '''
        embeddings = self.contextual_embeddings(entity_text)
        embeddings = embeddings.detach().clone().requires_grad_(True)
        normalized_embeddings = embeddings / torch.linalg.vector_norm(embeddings)

        return normalized_embeddings

    def prepare_data(self, data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns a tuple with lists of normalized embeddings and 
        similarity scores for the given dataset
        '''
        normalized_embeddings = []
        similarity_scores = []

        for _, row in data.iterrows():
            entity1 = row['Entity1']
            entity2 = row['Entity2']
            similarity = row['Similarity']

            embeddings1_normalized = self.get_normalized_embeddings(entity1)
            embeddings2_normalized = self.get_normalized_embeddings(entity2)

            normalized_embeddings.append(embeddings1_normalized)
            normalized_embeddings.append(embeddings2_normalized)
            similarity_scores.append(similarity)

        normalized_embeddings = torch.cat(normalized_embeddings)
        similarity_scores = torch.tensor(similarity_scores).view(-1, 1)

        return normalized_embeddings, similarity_scores
    

if __name__ == '__main__':

    MODEL_NAME = 'bert-base-uncased'
    data = pd.read_csv('data/dataset/train.csv')

    bert_embeddings = BERTEmbeddingsWithCLS(MODEL_NAME)
    data_prep = DataPreparation(MODEL_NAME, bert_embeddings)

    normalized_embeddings, similarity_scores = data_prep.prepare_data(data)

    print(normalized_embeddings.shape)
    print(similarity_scores.shape)

