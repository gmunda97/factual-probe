import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



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

    def _get_relation_counts(self, dataset_path):
        relation_counts = {}
        with open(dataset_path, 'r') as file:
            for line in file:
                if dataset_path.endswith('.jsonl'):
                    data = json.loads(line)
                    relation_id = data['pred_id']
                else:
                    parts = line.strip().split('\t')
                    relation_id = parts[1]
                if relation_id in relation_counts:
                    relation_counts[relation_id] += 1
                else:
                    relation_counts[relation_id] = 1

        counts = list(relation_counts.values())
        total_triples = sum(counts)
        normalized_counts = [count / total_triples for count in counts]

        print(f"Dataset: {dataset_path} - Relation Counts: {counts[:10]} - Normalized Counts: {normalized_counts[:10]}")  # Debugging
        return normalized_counts

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
        counts = self._get_relation_counts(self.file_path)

        plt.figure(figsize=(10, 8))
        sns.kdeplot(counts, fill=True, color='blue')
        plt.xlabel('Proportion of Triples per Relation')
        plt.ylabel('Density')
        plt.title('Relation Distribution KDE in the Reduced Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/reduced_relation_distribution_kde.png')

        plt.figure(figsize=(10, 8))
        sns.ecdfplot(counts, color='blue')
        plt.xlabel('Proportion of Triples per Relation')
        plt.ylabel('Cumulative Probability')
        plt.title('Relation Distribution CDF in the Reduced Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/reduced_relation_distribution_cdf.png')

    def plot_original_relation_distribution(self):
        if not self.original_dataset:
            print("Original dataset path is not set.")
            return

        counts = self._get_relation_counts(self.original_dataset)

        plt.figure(figsize=(10, 8))
        sns.kdeplot(counts, fill=True, color='red')
        plt.xlabel('Proportion of Triples per Relation')
        plt.ylabel('Density')
        plt.title('Relation Distribution KDE in the Original Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/original_relation_distribution_kde.png')

        plt.figure(figsize=(10, 8))
        sns.ecdfplot(counts, color='red')
        plt.xlabel('Proportion of Triples per Relation')
        plt.ylabel('Cumulative Probability')
        plt.title('Relation Distribution CDF in the Original Dataset')
        plt.grid(True)
        plt.savefig('./../resources/plots/original_relation_distribution_cdf.png')

    def plot_comparison(self):
        reduced_counts = self._get_relation_counts(self.file_path)
        original_counts = self._get_relation_counts(self.original_dataset)

        plt.figure(figsize=(6, 5))
        sns.kdeplot(reduced_counts, fill=True, color='blue', label='wikidata5m-sm')
        sns.kdeplot(original_counts, fill=True, color='red', label='wikidata5m')
        plt.xlabel('Proportion of Triples per Relation', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        #plt.title('Relation Distribution KDE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('./../resources/plots/relation_distribution_kde_comparison.png')

        plt.figure(figsize=(6, 5))
        sns.ecdfplot(reduced_counts, color='blue', label='wikidata5m-sm')
        sns.ecdfplot(original_counts, color='red', label='wikidata5m')
        plt.xlabel('Proportion of Triples per Relation', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        #plt.title('Relation Distribution CDF', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('./../resources/plots/relation_distribution_cdf_comparison.png')



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
    analyzer.plot_comparison()
