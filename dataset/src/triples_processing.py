'''
Script to retrieve entity information (textual values)
for the triples in the dataset using the Wikidata API.
'''

import requests
import json
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO)


class RateLimitException(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    wait=wait_exponential(multiplier=1, max=10),  # Wait exponentially between attempts, starting at 1s, max 10s
    retry=retry_if_exception_type(RateLimitException),  # Retry only if a RateLimitException is raised
)
def fetch_entity_info(wikidata_ids):
    wikidata_api_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': "|".join(wikidata_ids)
    }
    try:
        response = requests.get(wikidata_api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data['entities']
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise RateLimitException('Rate limit exceeded or server error')
        else:
            logging.error(f'HTTP Error fetching entity information: {e}')
            return {}
    except requests.exceptions.RequestException as e:
        logging.error(f'Request exception: {e}')
        return {}

def get_entity_info(wikidata_ids):
    entities_info = fetch_entity_info(wikidata_ids)
    return entities_info

def add_entity_values_batch(entries, batch_size=50):
    batches = [entries[i:i+batch_size] for i in range(0, len(entries), batch_size)]
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_entity_info, [
            entry['sub_id'], entry['pred_id'], entry['obj_id']
        ]): entry for batch in batches for entry in batch}

        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                results.append((entry, result))
            except Exception as exc:
                logging.error(f'Entity fetch generated an exception: {exc}. Entry: {entry}')

    return results

def process_chunk_and_save(entries, temp_dir, temp_file_prefix, chunk_index, include_entity_info=True):
    results = add_entity_values_batch(entries)

    temp_file_path = Path(temp_dir) / f"{temp_file_prefix}_chunk_{chunk_index}.jsonl"
    
    # Save results to a temporary file, including entity information if required
    with temp_file_path.open('w', encoding='utf-8') as temp_file:
        for entry, entities_info in results:
            if include_entity_info:
                # Extract and enrich entry with entity information
                sub_info = entities_info.get(entry['sub_id'], {})
                pred_info = entities_info.get(entry['pred_id'], {})
                obj_info = entities_info.get(entry['obj_id'], {})
                entry['sub_value'] = sub_info.get('labels', {}).get('en', {}).get('value', 'Unknown')
                entry['pred_value'] = pred_info.get('labels', {}).get('en', {}).get('value', 'Unknown')
                entry['obj_value'] = obj_info.get('labels', {}).get('en', {}).get('value', 'Unknown')
                temp_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return temp_file_path


def incremental_concatenate(temp_file_path, output_file):
    # Incrementally concatenate temp_file content to the output file
    with open(output_file, 'a', encoding='utf-8') as final_file:
        with open(temp_file_path, 'r', encoding='utf-8') as temp_file:
            for line in temp_file:
                final_file.write(line)
    # Optionally, delete the temporary file after its content has been appended
    Path(temp_file_path).unlink()

def txt_to_jsonl_with_incremental_concat(input_file, output_file, temp_dir='./temp', temp_file_prefix='wikidata', batch_size=50):
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as jsonl_file:
            entries = [json.loads(line) for line in jsonl_file]
    elif input_file.endswith('.tsv') or input_file.endswith('.txt'):
        with open(input_file, 'r', encoding='utf-8') as txt_file:
            lines = txt_file.readlines()
        entries = [{'sub_id': line.split('\t')[0], 'pred_id': line.split('\t')[1], 'obj_id': line.split('\t')[2].strip()} for line in lines]
    else:
        logging.error(f'Unsupported input file format: {input_file}')
        return
    
    # Ensure the temporary directory and output file are ready
    Path(temp_dir).mkdir(exist_ok=True)
    Path(output_file).unlink(missing_ok=True)  # Remove the output file if it exists

    # Process dataset in chunks, save to temporary files, and incrementally concatenate
    for i in tqdm(range(0, len(entries), batch_size), desc='Overall Progress'):
        chunk_entries = entries[i:i+batch_size]
        temp_file_path = process_chunk_and_save(chunk_entries, temp_dir, temp_file_prefix, i // batch_size)
        incremental_concatenate(temp_file_path, output_file)
    
    logging.info(f'Processed {len(entries)} entries and wrote to {output_file}')


if __name__ == '__main__':

    input_file = './../data/wikidata5m_inductive/wikidata5m_inductive_train_40k.tsv'
    output_file = './../data/wikidata5m_inductive/3wikidata5m_inductive_train_42k_sim_desc.jsonl'
    txt_to_jsonl_with_incremental_concat(input_file, output_file)