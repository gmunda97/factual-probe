'''
Script to fetch similarity scores for the triples in the dataset
using the WEmbedder API.
'''

import requests
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)



def get_object_embeddings(obj_id):
    api_url = f'https://wembedder.toolforge.org/api/vector/{obj_id}'
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('vector')
    except Exception:
        #print(f"Error fetching similarity for {sub_id} and {obj_id}: {e}")
        return None

def process_single_triple(triple):
    obj_embedding = get_object_embeddings(triple['obj_id'])
    if obj_embedding is not None:
        triple['obj_embedding'] = obj_embedding
    return triple

def process_triples_parallel(input_file, output_file, max_workers=10):
    triples_to_process = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        triples_to_process = [json.loads(line) for line in infile]

    # Using ThreadPoolExecutor to parallelize the API requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all triples for processing
        future_to_triple = {executor.submit(process_single_triple, triple): triple for triple in triples_to_process}
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for future in tqdm(as_completed(future_to_triple)):
                triple = future_to_triple[future]
                try:
                    updated_triple = future.result()
                    json.dump(updated_triple, outfile, ensure_ascii=False)
                    outfile.write('\n')
                except Exception as exc:
                    print(f'{triple} generated an exception: {exc}')


if __name__ == '__main__':
    input_file_path = './../../data/wikidata5m_inductive/wikidata5m_inductive_train_42k_sim_desc.jsonl'
    output_file_path = './../../data/wikidata5m_inductive/wikidata5m_inductive_train_42k_sim_desc_obj_embeddings.jsonl'
    process_triples_parallel(input_file_path, output_file_path)
