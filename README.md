## Model

How to unveil factual knowledge in PLMs vector spaces: Subjects and objects are inserted into a PLM to extract contextual embeddings out of them. The contextual embeddings $\mathbf{h}_s$ and $\mathbf{h}_o$ undergo a transformation by an additional model, which outputs transformed embeddings $\mathbf{t}_s$ and $\mathbf{t}_o$. A similarity score $Y$ between the transformed vectors is then computed and compared to the true similarity scores $\hat{Y}$ present in the dataset.

![Model](./assets/model.png)