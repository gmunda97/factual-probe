import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config = {
    'model_name': 'bert-base-uncased', # answerdotai/ModernBERT-base, bert-base-uncased
    'batch_size': 64,
    'num_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'cols': ['subject', 'pred_value', 'object', 'obj_embedding'],
    'data_paths': {
        'train': os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_train_relations_emb.csv'),
        'val':   os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_valid_relations_emb.csv'),
        'test':  os.path.join(_ROOT, 'data', 'dataset', 'wikidata5m_42k_test_relations_emb.csv'),
    },
    'cache_paths': {
        'train_h_s': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_train_rel_h_s_bert_cls_kge.pt'),
        'train_h_r': os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_train_rel_h_r_bert_cls_kge.pt'),
        'val_h_s':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_val_rel_h_s_bert_cls_kge.pt'),
        'val_h_r':   os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_val_rel_h_r_bert_cls_kge.pt'),
        'test_h_s':  os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_test_rel_h_s_bert_cls_kge.pt'),
        'test_h_r':  os.path.join(_ROOT, 'data', 'embeddings', 'relational_probing', '42k_test_rel_h_r_bert_cls_kge.pt'),
    },
    'model_paths': {
        'saved_model': os.path.join(_ROOT, 'trained_models', 'relational_probing', 'transform_triples_kge_bert_cls.pt'),
        'loss_curve':  './../resources/plots/bert/loss_curve_kge_bert_cls.png',
    },
    'results_paths': {
        'evaluation_results': os.path.join(_ROOT, 'data', 'results', 'relational_probing', 'evaluation_results_kge_bert_cls.xlsx'),
    },
}


def get_config():
    return config
