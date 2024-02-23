import requests
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)


def fetch_entity_info(wikidata_ids):
    wikidata_api_url = "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids=" + "|".join(wikidata_ids)
    try:
        response = requests.get(wikidata_api_url)
        response.raise_for_status()
        data = response.json()
        return data['entities']
    except Exception as e:
        print(f"Error fetching entity information: {e}")
        return {}

def get_entity_info(wikidata_ids):
    entities_info = fetch_entity_info(wikidata_ids)
    return entities_info

def add_entity_values_batch(entries, batch_size=50):
    # Split entries into batches
    batches = [entries[i:i+batch_size] for i in range(0, len(entries), batch_size)]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for batch in tqdm(batches[:30], desc="Processing batches"):
            futures = []
            for entry in batch:
                wikidata_ids = [entry['sub_id'], entry['pred_id'], entry['obj_id']]
                futures.append(executor.submit(get_entity_info, wikidata_ids))
            for future in futures:
                result = future.result()
                results.append(result)

    return results

def txt_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    jsonl_entries = []
    for line in lines:
        parts = line.strip().split('\t')
        # Extract subject, predicate, and object
        sub_id, pred_id, obj_id = parts
        entry = {
            'sub_id': sub_id,
            'pred_id': pred_id,
            'obj_id': obj_id
        }
        jsonl_entries.append(entry)

    # Batch processing of entries
    results = add_entity_values_batch(jsonl_entries)

    # Combine results and write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for entry, entities_info in zip(jsonl_entries, results):
            sub_info = entities_info.get(entry['sub_id'], {})
            pred_info = entities_info.get(entry['pred_id'], {})
            obj_info = entities_info.get(entry['obj_id'], {})

            entry['sub_value'] = sub_info.get('labels', {}).get('en', {}).get('value')
            entry['pred_value'] = pred_info.get('labels', {}).get('en', {}).get('value')
            entry['obj_value'] = obj_info.get('labels', {}).get('en', {}).get('value')

            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')


input_file = './../data/wikidata5m_inductive/wikidata5m_inductive_train_40k.tsv'
output_file = './../data/wikidata5m_inductive/wikidata5m_inductive_train_40k.jsonl'
txt_to_jsonl(input_file, output_file)