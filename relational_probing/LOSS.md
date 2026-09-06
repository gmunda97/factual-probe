# InfoNCE Loss

## 1. Simple explanation

InfoNCE is a loss for learning which item belongs with which other item.

For each training example, the model produces a predicted object embedding `o_hat`. We know the correct object embedding `o`. InfoNCE asks the model to:

- give a high similarity score to the correct object;
- give lower similarity scores to the other objects in the same batch.

The correct pair is called the **positive pair**. The other candidate pairs are **negative pairs**.

In this project, the model predicts an object from the subject and relation:

```text
h_o_hat = W(h_s + h_r)
```

InfoNCE encourages `h_o_hat` to be close to its correct object embedding `h_o` and farther from the object embeddings belonging to the other triples in the batch.

A useful intuition is that cosine loss asks:

> Is the prediction close to its target?

InfoNCE additionally asks:

> Is the correct target more similar than the alternatives?

This makes InfoNCE a ranking or contrastive loss, not only a pairwise similarity loss.

## 2. The basic computation

Suppose a batch contains `N` examples. Let:

- `q_i` be the predicted embedding for example `i`;
- `k_j` be the true object embedding for example `j`;
- `sim(q_i, k_j)` be their similarity.

The positive object for prediction `q_i` is `k_i`. The objects `k_j` for `j != i` are the in-batch negatives.

The InfoNCE loss for example `i` is:

$$
L_i = -\log
\frac{\exp(sim(q_i, k_i)/T)}
{\sum_{j=1}^{N} \exp(sim(q_i, k_j)/T)}
$$

The batch loss is the average:

$$
L_{NCE} = \frac{1}{N}\sum_{i=1}^{N} L_i
$$

Here, `T` is the **temperature**. Lower temperatures make the model focus more strongly on the difference between the most similar candidates.

In the code, the similarity is cosine similarity because both embedding matrices are normalized first:

```python
q = F.normalize(o_hat, dim=1)
k = F.normalize(o, dim=1)
logits = (q @ k.T) / temperature
```

The matrix `q @ k.T` has shape `(N, N)`. Entry `(i, j)` is the similarity between prediction `i` and target `j`. The correct class for row `i` is column `i`, so the labels are:

```python
labels = torch.arange(len(q))
```

PyTorch's cross-entropy loss then applies the softmax and computes the formula above.

## 3. InfoNCE in this project

The model first builds an additive query:

$$
\hat{h}_o = h_s + h_r
$$

The linear transformation produces the predicted object embedding:

$$
q = W\hat{h}_o = W(h_s + h_r)
$$

The target is the BERT embedding of the true object:

$$
k = h_o
$$

For a minibatch, the loss compares every predicted object with every true object in that minibatch. The diagonal entries are the intended positive pairs:

```text
             target 0   target 1   target 2
query 0       positive   negative   negative
query 1       negative   positive   negative
query 2       negative   negative   positive
```

The current combined training objective is:

$$
L = L_{cos} + \lambda L_{NCE}
$$

where:

$$
L_{cos} = \frac{1}{N}\sum_i \left(1 - \cos(q_i, k_i)\right)
$$

and `lambda` controls the strength of the ranking objective.

In the current configuration:

```python
'use_infonce': True,
'infonce_lambda': 1.0,
'infonce_temperature': 0.05,
'batch_size': 264,
```

Therefore, a full batch has 264 candidates per query: one positive and 263 in-batch negatives. The final batch may be smaller, so it has fewer negatives.

If the batch size is set to 512, each full batch has 511 in-batch negatives. This is why batch size is especially important for InfoNCE: it changes the difficulty and candidate pool of the ranking problem.

## 4. Role of the temperature

The temperature rescales the similarities before the softmax:

$$
logit_{ij} = \frac{sim(q_i, k_j)}{T}
$$

- A larger `T` produces softer probabilities and weaker distinctions between candidates.
- A smaller `T` produces sharper probabilities and penalizes confusing hard negatives more strongly.

The project currently uses `T = 0.05`, which is relatively sharp. This can improve ranking pressure, but it can also make training sensitive to noisy or false negatives.

## 5. Role of lambda

The combined objective is:

$$
L = L_{cos} + \lambda L_{NCE}
$$

- `lambda = 0` makes the objective cosine-only.
- A small positive `lambda` gives InfoNCE a supporting role.
- `lambda = 1` gives the raw cosine and InfoNCE terms equal coefficient, although their numerical scales may still differ.
- A large `lambda` makes the model prioritize ranking over direct pairwise alignment.

The optional implementation is controlled by:

```python
'use_infonce': False
```

When it is `False`, the loss is only `cosine_loss`. When it is `True`, the InfoNCE term is added.

## 6. Why minibatches matter

With in-batch negatives, the batch is also the candidate set. For batch size `N`, each example normally has `N - 1` negatives:

| Batch size | Negatives per example |
|---:|---:|
| 64 | 63 |
| 256 | 255 |
| 264 | 263 |
| 512 | 511 |
| 1024 | 1023 |

Larger batches provide more negatives and usually a more informative ranking problem. They also use more memory and produce fewer optimizer updates per epoch.

For this project, embeddings are precomputed, so the main memory cost comes from the tensors used by the linear model and the `(N, N)` similarity matrix inside InfoNCE. A batch of 512 requires a `512 x 512` logits matrix for each loss calculation.

## 7. Important caveat: false negatives

InfoNCE assumes that the other targets in the batch are negative for a given query. This assumption is not always true in a knowledge graph.

For example, two different triples may have the same object:

```text
(subject A, relation, object X)
(subject B, relation, object X)
```

If both examples occur in the same batch, the embedding of `object X` is positive for both queries. However, the standard diagonal formulation treats the other occurrence as a negative for each query. This is a **false negative**.

False negatives can become more common as the batch size grows. They are a reason to compare batch sizes such as 256 and 512 rather than assuming that the largest possible batch is best.

Possible remedies include:

- masking other occurrences of the same object from the denominator;
- using a multi-positive contrastive loss;
- grouping or sampling batches to control duplicate objects;
- keeping the batch size moderate.

The current implementation uses the standard diagonal-label formulation and does not yet mask duplicate objects.

## 8. What the loss optimizes

Cosine loss aligns each prediction with its own target. InfoNCE adds relative separation:

```text
positive similarity > negative similarities
```

A model can have a good average positive cosine similarity but still rank an incorrect object above the correct one. InfoNCE directly addresses this ranking behavior.

Conversely, a model can improve its ranking loss by separating candidates while not producing the best possible absolute alignment. Combining both terms is intended to balance these goals:

- cosine loss: absolute pairwise alignment;
- InfoNCE: relative discrimination among candidates.

## 9. Practical checks

When comparing configurations, monitor both the training objective and retrieval metrics such as MRR and Hits@1. Useful experiments include:

```python
batch_size = 256 or 512
temperature = 0.05 or 0.1
infonce_lambda = 0.25, 0.5, or 1.0
```

Keep the data split, cached embeddings, optimizer settings, and random seed fixed when comparing runs. Because the current loss is batch-dependent, changing the batch size changes the objective as well as the optimization dynamics.
