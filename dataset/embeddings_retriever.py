import numpy as np

def load_embeddings(entity_embeddings_path='entity_embeddings.npy', relation_embeddings_path='relation_embeddings.npy'):
    """Load saved embeddings from .npy files."""
    entity_embeddings = np.load(entity_embeddings_path, allow_pickle=True)
    relation_embeddings = np.load(relation_embeddings_path, allow_pickle=True)
    return entity_embeddings, relation_embeddings

def get_entity_to_id_mapping(entity_to_id_path='entity_to_id.npy'):
    """Load the saved entity_to_id mapping."""
    return np.load(entity_to_id_path, allow_pickle=True).item()

def get_specific_entity_embedding(entity_name, entity_to_id, entity_embeddings):
    """Retrieve the embedding for a specific entity by name."""
    entity_id = entity_to_id.get(entity_name)
    if entity_id is not None:
        return entity_embeddings[entity_id]
    else:
        print(f"Entity '{entity_name}' not found.")
        return None

if __name__ == "__main__":
    # Load embeddings and entity_to_id mapping
    entity_embeddings, relation_embeddings = load_embeddings()
    entity_to_id = get_entity_to_id_mapping()
    
    # Example: Retrieve and print the embedding for a specific entity
    specific_entity_name = "Susan Perkins"  # Replace with the actual entity name you're interested in
    # Ensure the specific entity name matches exactly with how it's stored in your dataset/entity_to_id mapping
    
    specific_entity_embedding = get_specific_entity_embedding(specific_entity_name, entity_to_id, entity_embeddings)
    
    if specific_entity_embedding is not None:
        print(f"Embedding for '{specific_entity_name}': {specific_entity_embedding}")
    else:
        print(f"Could not retrieve embedding for '{specific_entity_name}'. Make sure the entity name exactly matches one in the dataset.")