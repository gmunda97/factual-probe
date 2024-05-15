config = {
    'model_name': 'facebook/bart-base', # bert-base-uncased, openai-community/gpt2, facebook/bart-base
    'embedding_dim': 768,
    'hidden_dim': 512,
    'rbf_features': 100,
    'learning_rate': 0.005,
    'weight_decay': 0.005,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'optimizer': 'AdamW',  # Could be SGD, Adam, etc.
    'loss_function': 'MSELoss',
    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'mode': 'min',
        'factor': 0.1,
        'patience': 5,
    },
    'data_paths': {
        'train_data': './../../data/dataset/wikidata5m_42k_desc_train.csv',
        'val_data': './../../data/dataset/wikidata5m_42k_desc_valid.csv',
        'train_embeddings': './../../data/embeddings/wikidata5m_42k_desc_train_embeddings_bart.pt',
        'val_embeddings': './../../data/embeddings/wikidata5m_42k_desc_valid_embeddings_bart.pt',
    },
    'model_paths': {
        'saved_model_full_dim': './../../trained_models/wiki_desc/full_dim/best_model.pth',
        'saved_model_reduced_dim': './../../trained_models/wiki_desc/reduced_dim/best_model',
    },
}

def get_config():
    return config