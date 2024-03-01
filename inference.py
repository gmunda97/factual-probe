import torch
import torch.nn.functional as F

from bert_embeddings import CreateBERTEmbeddingsWithCLS


entiy1 = "All This, and Heaven Too"
entity2 = "Jeffrey Lynn"

embedding_entity1 = CreateBERTEmbeddingsWithCLS('bert-base-uncased')(entiy1)
embedding_entity2 = CreateBERTEmbeddingsWithCLS('bert-base-uncased')(entity2)

EMBEDDING_DIM = embedding_entity1.shape[1]


checkpoint = torch.load('models/linear_transformation.pth')
# load the weights into my linear model
loaded_linear_transformation = checkpoint['model_class'](EMBEDDING_DIM, EMBEDDING_DIM)
# loading the state dictionary
loaded_linear_transformation.load_state_dict(checkpoint['state_dict'])
loaded_linear_transformation.eval()


def get_transformed_similarity(entity1_embedding: torch.Tensor, entity2_embedding: torch.Tensor) -> float:
    transformed_entity1 = loaded_linear_transformation(entity1_embedding)
    transformed_entity2 = loaded_linear_transformation(entity2_embedding)

    similarity_score = F.cosine_similarity(transformed_entity1, transformed_entity2, dim=1).item()

    return similarity_score

def get_plain_similarity(entity1_embedding: torch.Tensor, entity2_embedding: torch.Tensor) -> float:
    similarity_score = F.cosine_similarity(entity1_embedding, entity2_embedding, dim=1).item()

    return similarity_score


print(get_transformed_similarity(embedding_entity1, embedding_entity2))
print(get_plain_similarity(embedding_entity1, embedding_entity2))