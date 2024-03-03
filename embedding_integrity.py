import torch
import pandas as pd

from dataload_batches import DataPreparation
from bert_embeddings import BERTEmbeddings, BERTEmbeddingsWithCLS


def load_embeddings_and_scores_from_torch(file_path: str):
    loaded_data = torch.load(file_path)
    embeddings = loaded_data['embeddings']
    similarity_scores = loaded_data['similarity_scores']
    return embeddings, similarity_scores

train_data = pd.read_csv('./data/dataset/wikidata5m_6k_train.csv')
val_data = pd.read_csv('data/dataset/wikidata5m_6k_valid.csv')

MODEL_NAME = 'bert-base-uncased'
EMBEDDINGS_PYTORCH_TRAIN = './data/embeddings/2wikidata5m_6k_train_embeddings.pt'
EMBEDDINGS_PYTORCH_VAL = './data/embeddings/3wikidata5m_6k_valid_embeddings.pt'

#saved_train_embeddings = torch.load(EMBEDDINGS_PYTORCH_TRAIN)
saved_val_embeddings, saved_sim_scores = load_embeddings_and_scores_from_torch(EMBEDDINGS_PYTORCH_VAL)

bert_embeddings = BERTEmbeddingsWithCLS(MODEL_NAME)
data_prep = DataPreparation(MODEL_NAME, bert_embeddings)
#normalized_embeddings_train, similarity_scores_train = data_prep.prepare_data(train_data)
normalized_embeddings_val, similarity_scores_val = data_prep.prepare_data(val_data)


assert torch.allclose(saved_val_embeddings[:10], normalized_embeddings_val[:10], atol=1e-6), "Embeddings do not match"
assert torch.allclose(saved_sim_scores[:10], similarity_scores_val[:10], atol=1e-6), "Similarity scores do not match"