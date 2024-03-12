import pandas as pd
import torch
import torch.nn.functional as F

from utils import UtilityFunctions


test_data = pd.read_csv('./../../data/dataset/wikidata5m_42k_test.csv')
MODEL_NAME = 'bert-base-uncased'

EMBEDDINGS_PYTORCH_TEST = './../../data/embeddings/wikidata5m_42k_test_embeddings.pt'
utils = UtilityFunctions()

saved_test_embeddings, similarity_scores_test = utils.load_embeddings_and_scores_from_torch(EMBEDDINGS_PYTORCH_TEST)
normalized_embeddings_test = saved_test_embeddings

EMBEDDING_DIM = normalized_embeddings_test.shape[1]

trained_model = './../../trained_models/42k_linear_no_bias.pth'
checkpoint = torch.load(trained_model)

if 'orthogonal' in trained_model:
    loaded_transformation = checkpoint['model_class'](EMBEDDING_DIM)
else:
    # load the weights into the model
    loaded_transformation = checkpoint['model_class'](EMBEDDING_DIM, EMBEDDING_DIM)

# loading the state dictionary
loaded_transformation.load_state_dict(checkpoint['state_dict'])
loaded_transformation.eval()

transformed_embeddings = loaded_transformation(normalized_embeddings_test)
predicted_similarity_scores = F.cosine_similarity(
    transformed_embeddings[::2],
    transformed_embeddings[1::2],
    dim=1
).view(-1, 1)

person_correlation = utils.compute_pearson_correlation(predicted_similarity_scores, similarity_scores_test)
spearman_correlation = utils.compute_spearman_correlation(predicted_similarity_scores, similarity_scores_test)
mean_squared_error = utils.compute_mean_squared_error(predicted_similarity_scores, similarity_scores_test)
root_mean_squared_error = utils.compute_root_mean_squared_error(predicted_similarity_scores, similarity_scores_test)

print(f"Person correlation: {person_correlation}")
print(f"Spearman correlation: {spearman_correlation}")
print(f"Mean squared error: {mean_squared_error}")
print(f"Root mean squared error: {root_mean_squared_error}")