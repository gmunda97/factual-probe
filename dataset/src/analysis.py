import json
import matplotlib.pyplot as plt



class KnowledgeGraphAnalyzer:
    def __init__(self, file_path: str, original_dataset: str = None):
        self.file_path = file_path
        self.original_dataset = original_dataset

    def _read_dataset(self, unique_key_extractor):
        unique_keys = set()
        with open(self.file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                keys = unique_key_extractor(data)
                if isinstance(keys, tuple):
                    unique_keys.update(keys)
                else:
                    unique_keys.add(keys)
        return len(unique_keys)

    def count_unique_entities(self):
        return self._read_dataset(lambda data: (data['sub_id'], data['obj_id']))

    def count_unique_relations(self):
        return self._read_dataset(lambda data: data['pred_id'])

    def count_unique_objects(self):
        return self._read_dataset(lambda data: data['obj_id'])

    def count_unique_subjects(self):
        return self._read_dataset(lambda data: data['sub_id'])

    def count_unique_triples(self):
        return self._read_dataset(
            lambda data: (data['sub_id'], data['pred_id'], data['obj_id'])
        )

    def plot_relation_distribution(self):
        relation_counts = {}
        with open(self.file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data['pred_id'] in relation_counts:
                    relation_counts[data['pred_id']] += 1
                else:
                    relation_counts[data['pred_id']] = 1

        counts = list(relation_counts.values())

        plt.figure(figsize=(10, 8))
        plt.hist(counts, bins=20, color='green', edgecolor='black')
        plt.xlabel('Number of Triples per Relation')
        plt.ylabel('Number of Relations')
        plt.title('Relation Distribution in the Reduced Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/relation_distribution.png')

    def plot_original_relation_distribution(self):
        if not self.original_dataset:
            print("Original dataset path is not set.")
            return

        relation_counts = {}
        with open(self.original_dataset, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                relation_id = parts[1]
                if relation_id in relation_counts:
                    relation_counts[relation_id] += 1
                else:
                    relation_counts[relation_id] = 1

        counts = list(relation_counts.values())

        plt.figure(figsize=(10, 8))
        plt.hist(counts, bins=20, color='red', edgecolor='black')
        plt.xlabel('Number of Triples per Relation')
        plt.ylabel('Number of Relations')
        plt.title('Relation Distribution in the Original Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/original_relation_distribution.png')


if __name__ == '__main__':

    file_path = './../../data/wikidata5m_inductive/wikidata5m_inductive_train_42k_sim_desc.jsonl'
    original_dataset = './../../data/wikidata5m_inductive/wikidata5m_inductive_train.txt'

    analyzer = KnowledgeGraphAnalyzer(file_path, original_dataset)
    print("Unique Entities:", analyzer.count_unique_entities())
    print("Unique Relations:", analyzer.count_unique_relations())
    print("Unique Objects:", analyzer.count_unique_objects())
    print("Unique Subjects:", analyzer.count_unique_subjects())
    print("Unique Triples:", analyzer.count_unique_triples())
    analyzer.plot_relation_distribution()
    analyzer.plot_original_relation_distribution()

