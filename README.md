## Factual Probe

An innovative method that uses probing to unveil implicit factual knowledge present in Pre-trained Language Models (PLMs) representation space.

### Method

A dataset consisting of knowledge graph triples and a similarity score between the subject and object pairs is collected. The dataset is constructed from the Wikidata knowledge graph using the Wikidata5m dataset (https://deepgraphlearning.github.io/project/wikidata5m).

Subjects and objects from the knowledge graph are inserted into a PLM to extract contextual embeddings out of them. The contextual embeddings $\mathbf{h}_s$ and $\mathbf{h}_o$ undergo a transformation by an additional model, which outputs transformed embeddings $\mathbf{t}_s$ and $\mathbf{t}_o$. A similarity score $Y$ between the transformed vectors is then computed and compared to the true similarity scores $\hat{Y}$ present in the dataset. The hypothesis is that factual knowledge is embedded in the language model if the linear transformation is able to reproduce the similarity scores of the knowledge graph embeddings using only the contextual embeddings.

![Model](./assets/model.png)