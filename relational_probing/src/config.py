import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_SPEC = "bert_layer11_infonce"
EMBEDDING_SPEC = "bert_layer11"

config = {
    'model_name': 'bert-base-uncased', # answerdotai/ModernBERT-base, bert-base-uncased
    'transformer_layer': 11,
    'batch_size': 264,  # 64 (old implementation before InfoNCE) 512 or 264 (new implementation with InfoNCE)
    'num_epochs': 60,  # 100 (old implementation before InfoNCE) 60 (new implementation with InfoNCE)
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 12,  # 10 (old implementation before InfoNCE) 12 (new implementation with InfoNCE)
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'use_infonce': True,
    'infonce_lambda': 1.0,
    'infonce_temperature': 0.05,
    'load_balance_weight': 0.01,
    'cols': ['subject', 'pred_value', 'object'], # add 'obj_embedding' if using KGE embedding dataset
    'data_paths': {
        'train': os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_train_relations.csv'),
        'val':   os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_valid_relations.csv'),
        'test':  os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_test_relations.csv'),
    },
    'cache_paths': {
        'train_h_s': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_train_rel_h_s_{EMBEDDING_SPEC}.pt'),
        'train_h_r': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_train_rel_h_r_{EMBEDDING_SPEC}.pt'),
        'train_h_o': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_train_rel_h_o_{EMBEDDING_SPEC}.pt'),
        'val_h_s':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_val_rel_h_s_{EMBEDDING_SPEC}.pt'),
        'val_h_r':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_val_rel_h_r_{EMBEDDING_SPEC}.pt'),
        'val_h_o':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_val_rel_h_o_{EMBEDDING_SPEC}.pt'),
        'test_h_s':  os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_test_rel_h_s_{EMBEDDING_SPEC}.pt'),
        'test_h_r':  os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_test_rel_h_r_{EMBEDDING_SPEC}.pt'),
        'test_h_o':  os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', f'42k_test_rel_h_o_{EMBEDDING_SPEC}.pt'),
    },
    'model_paths': {
        'saved_model': os.path.join(_ROOT, 'trained_models', 'relational_probing', f'transform_triples_{MODEL_SPEC}.pt'),
        'loss_curve':  f'./../resources/plots/bert/loss_curve_{MODEL_SPEC}.png',
    },
    'results_paths': {
        'evaluation_results': os.path.join(_ROOT, 'results', 'relational_probing', f'evaluation_results_{MODEL_SPEC}.xlsx'),
        'run_metadata_dir': os.path.join(_ROOT, 'results', 'relational_probing', 'runs'),
    },
}


def get_config():
    return config
