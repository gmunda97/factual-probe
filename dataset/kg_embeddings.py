import json
import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

def load_training_set(file_path):
    """Load the dataset from a JSONL file and return a list of triples."""
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            triple_data = json.loads(line)
            triples.append((triple_data['sub_value'], triple_data['pred_value'], triple_data['obj_value']))
    return np.array(triples)

def load_test_set(file_path):
    """Load dataset from a JSONL file."""
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            triple_data = json.loads(line)
            triples.append((triple_data['sub_value'], triple_data['pred_value'], triple_data['obj_value']))
    return np.array(triples)

def main(train_dataset_path, test_dataset_path):
    # Load the datasets
    train_triples = load_training_set(train_dataset_path)
    test_triples = load_test_set(test_dataset_path)
    
    # Create triples factories
    train_tf = TriplesFactory.from_labeled_triples(train_triples)
    test_tf = TriplesFactory.from_labeled_triples(test_triples, entity_to_id=train_tf.entity_to_id, relation_to_id=train_tf.relation_to_id)
    
    # Initialize the TransE model
    model = TransE(
        triples_factory=train_tf,
        embedding_dim=50,
        random_seed=42
    )
    
    # Initialize the training loop
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=train_tf,
        optimizer=torch.optim.Adam(params=model.parameters(), lr=0.001)
    )
    
    # Train the model
    training_loop.train(triples_factory=train_tf, num_epochs=1, batch_size=256)
    
    # Evaluate the model
    evaluator = RankBasedEvaluator()
    # Add the training triples for filtering
    results = evaluator.evaluate(
        model, 
        mapped_triples=test_tf.mapped_triples, 
        additional_filter_triples=[train_tf.mapped_triples], 
        batch_size=256
    )
    
    print(results)

    # Access and save entity embeddings
    entity_embeddings = model.entity_representations[0]()._tensor.detach().numpy()
    np.save('entity_embeddings.npy', entity_embeddings)
    
    # Access and save relation embeddings
    relation_embeddings = model.relation_representations[0]()._tensor.detach().numpy()
    np.save('relation_embeddings.npy', relation_embeddings)

    # Save entity_to_id mapping
    np.save('entity_to_id.npy', train_tf.entity_to_id)

if __name__ == "__main__":
    training_set = './../data/wikidata5m_inductive/wikidata5m_inductive_train_40k.jsonl'
    test_set = './../data/wikidata5m_inductive/wikidata5m_inductive_test.jsonl'
    main(training_set, test_set)
