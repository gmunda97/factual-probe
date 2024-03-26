'''Module to process the dataset for the model training'''

import os
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS, GPTEmbeddings, BARTEmbeddings


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['subject'], row['object'], row['similarity']


class DataPreparation:
    def __init__(self, model_name: str, contextual_embeddings: torch.Tensor) -> None:
        self.model_name = model_name
        self.contextual_embeddings = contextual_embeddings

    def get_normalized_embeddings(self, entity_text: str) -> torch.Tensor:
        '''
        Method to L2 normalize embeddings for the given entity text
        '''
        embeddings = self.contextual_embeddings(entity_text)
        embeddings = embeddings.detach()#.clone().requires_grad_(True)
        norm = torch.linalg.norm(embeddings, dim=1, keepdim=True)
        norm[norm == 0] = 1 # prevent division by zero
        normalized_embeddings = embeddings / norm

        return normalized_embeddings

    def prepare_batch(self, batch_data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Prepares a batch of normalized embeddings and similarity scores
        '''
        subjects, objects, similarities = batch_data
        batch_normalized_embeddings = []
        batch_similarity_scores = []

        for subject, obj in zip(subjects, objects):
            embeddings1_normalized = self.get_normalized_embeddings(subject)
            embeddings2_normalized = self.get_normalized_embeddings(obj)
            batch_normalized_embeddings.append(embeddings1_normalized)
            batch_normalized_embeddings.append(embeddings2_normalized)
        
        batch_normalized_embeddings = torch.cat(batch_normalized_embeddings, dim=0)
        batch_similarity_scores = torch.as_tensor(similarities, dtype=torch.float32).view(-1, 1)

        return batch_normalized_embeddings, batch_similarity_scores

    def prepare_data(self, data: pd.DataFrame, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns a tuple with lists of normalized embeddings and 
        similarity scores for the given dataset
        '''
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        dataset = CustomDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        normalized_embeddings = []
        similarity_scores = []

        for batch_data in tqdm(dataloader):
            batch_normalized_embeddings, batch_similarity_scores = self.prepare_batch(batch_data)
            normalized_embeddings.append(batch_normalized_embeddings)
            similarity_scores.append(batch_similarity_scores)
        
        normalized_embeddings = torch.cat(normalized_embeddings)
        similarity_scores = torch.cat(similarity_scores)

        return normalized_embeddings, similarity_scores
    

class DataOnDisk(DataPreparation):
    def __init__(self, model_name: str, contextual_embeddings: torch.Tensor, save_path: str) -> None:
        super().__init__(model_name, contextual_embeddings)
        self.save_path = save_path

    def _save_data_to_torch(self, embeddings: torch.Tensor, scores: torch.Tensor) -> None:
        data_to_save = {'embeddings': embeddings, 'similarity_scores': scores}
        torch.save(data_to_save, self.save_path)

    def prepare_data_and_save(self, data: pd.DataFrame, batch_size: int = 32) -> None:
        normalized_embeddings, similarity_scores = self.prepare_data(data, batch_size)
        self._save_data_to_torch(normalized_embeddings, similarity_scores)

        return normalized_embeddings, similarity_scores



if __name__ == '__main__':

    MODEL_NAME = 'bert-base-uncased'
    SAVE_PATH = './../../data/embeddings/wikidata5m_42k_valid_embeddings_bert.pt'
    #train_data = pd.read_csv('./../../data/dataset/wikidata5m_42k_train.csv')
    val_data = pd.read_csv('./../../data/dataset/wikidata5m_42k_valid.csv')
    #test_data = pd.read_csv('./../../data/dataset/wikidata5m_42k_test.csv')

    if os.path.exists(SAVE_PATH):
        bert_embeddings = BERTEmbeddingsWithCLS(MODEL_NAME)
        data_prep = DataPreparation(MODEL_NAME, bert_embeddings)
        normalized_embeddings, similarity_scores = data_prep.prepare_data(val_data)
        print(normalized_embeddings.shape)
        print(similarity_scores.shape)
        print(normalized_embeddings[0])

    else:
        bert_embeddings = BERTEmbeddings(MODEL_NAME)
        data_prep = DataOnDisk(MODEL_NAME, bert_embeddings, SAVE_PATH)
        normalized_embeddings, _ = data_prep.prepare_data_and_save(val_data)
