'''
Script to analyze the knowledge graph dataset after the splitting
in training, validation, and test sets.
'''

import json
import csv

class KnowledgeGraphAnalyzer:
    def __init__(self, file_path: str, original_dataset: str = None):
        self.file_path = file_path
        self.original_dataset = original_dataset

    def _read_dataset(self, unique_key_extractor):
        unique_keys = set()
        if self.file_path.endswith('.jsonl'):
            with open(self.file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    keys = unique_key_extractor(data)
                    if isinstance(keys, tuple):
                        unique_keys.update(keys)
                    else:
                        unique_keys.add(keys)
        elif self.file_path.endswith('.csv'):
            with open(self.file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    keys = unique_key_extractor(row)
                    if isinstance(keys, tuple):
                        unique_keys.update(keys)
                    else:
                        unique_keys.add(keys)
        return len(unique_keys)

    def _get_relation_counts(self, dataset_path):
        relation_counts = {}
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    relation_id = data['pred_id']
                    if relation_id in relation_counts:
                        relation_counts[relation_id] += 1
                    else:
                        relation_counts[relation_id] = 1
        elif dataset_path.endswith('.csv'):
            with open(dataset_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    relation_id = row['pred_value']
                    if relation_id in relation_counts:
                        relation_counts[relation_id] += 1
                    else:
                        relation_counts[relation_id] = 1

        counts = list(relation_counts.values())
        total_triples = sum(counts)
        normalized_counts = [count / total_triples for count in counts]

        print(f'Dataset: {dataset_path} - Relation Counts: {counts[:10]} - Normalized Counts: {normalized_counts[:10]}')
        return normalized_counts

    def count_unique_entities(self):
        return self._read_dataset(lambda data: (data['sub_id'], data['obj_id']) if 'sub_id' in data and 'obj_id' in data else (data['subject'], data['object']))

    def count_unique_relations(self):
        return self._read_dataset(lambda data: data['pred_id'] if 'pred_id' in data else data['pred_value'])

    def count_unique_objects(self):
        return self._read_dataset(lambda data: data['obj_id'] if 'obj_id' in data else data['object'])

    def count_unique_subjects(self):
        return self._read_dataset(lambda data: data['sub_id'] if 'sub_id' in data else data['subject'])

    def count_unique_triples(self):
        return self._read_dataset(
            lambda data: (data['sub_id'], data['pred_id'], data['obj_id']) if 'sub_id' in data and 'pred_id' in data and 'obj_id' in data else (data['subject'], data['pred_value'], data['object'])
        )

if __name__ == '__main__':
    
    file_path_jsonl = 'path_to_your_jsonl_file.jsonl'
    file_path_csv = './../../data/dataset/wikidata5m_42k_valid_relations.csv'

    #analyzer_jsonl = KnowledgeGraphAnalyzer(file_path_jsonl)
    analyzer = KnowledgeGraphAnalyzer(file_path_csv)

    print('Unique Entities:', analyzer.count_unique_entities())
    print('Unique Relations:', analyzer.count_unique_relations())
    print('Unique Objects:', analyzer.count_unique_objects())
    print('Unique Subjects:', analyzer.count_unique_subjects())
    print('Unique Triples:', analyzer.count_unique_triples())
