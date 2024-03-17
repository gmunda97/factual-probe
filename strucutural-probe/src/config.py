config = {
    'model_name': 'bert-base-uncased',
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
        'train_data': './../data/dataset/wikidata5m_6k_train.csv',
        'val_data': './../data/dataset/wikidata5m_6k_valid.csv',
        'train_embeddings': './../data/embeddings/3wikidata5m_6k_train_embeddings.pt',
        'val_embeddings': './../data/embeddings/3wikidata5m_6k_valid_embeddings.pt',
    },
    'model_paths': {
        'saved_model': './../trained_models/best_model.pth'
    },
}

def get_config():
    return config