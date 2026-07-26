import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config = {
    'model_name': 'answerdotai/ModernBERT-base', # answerdotai/ModernBERT-base, bert-base-uncased
    'batch_size': 64,
    'num_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'cols': ['subject', 'pred_value', 'object'],
    'data_paths': {
        'train': os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_train_relations.csv'),
        'val':   os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_valid_relations.csv'),
    },
    'cache_paths': {
        'train_h_s': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_train_rel_h_s_modernbert.pt'),
        'train_h_r': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_train_rel_h_r_modernbert.pt'),
        'train_h_o': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_train_rel_h_o_modernbert.pt'),
        'val_h_s':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_val_rel_h_s_modernbert.pt'),
        'val_h_r':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_val_rel_h_r_modernbert.pt'),
        'val_h_o':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_val_rel_h_o_modernbert.pt'),
    },
    'model_paths': {
        'saved_model': os.path.join(_ROOT, 'trained_models', 'relational_probing', 'transform_triples_modernbert.pt'),
        'loss_curve':  './../resources/plots/bert/loss_curve_modernbert.png',
    },
}


def get_config():
    return config
