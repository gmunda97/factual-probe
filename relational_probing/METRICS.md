# Evaluation Metrics — Relational Probing

This document explains the three metrics used to evaluate the triples-completion
experiment, using real examples drawn from the Wikidata5M 42k test split.

---

## The task in one sentence

Given a KG triple *(subject, relation, object)*, encode the subject and relation
with BERT, add the vectors together, optionally apply a learned linear
transformation **W**, and measure how well the resulting vector retrieves the
correct object from a pool of all test objects.

```
h_s  =  BERT("Aarhus University")
h_r  =  BERT("instance of")
ĥ    =  h_s + h_r          ← additive prediction
Wĥ   =  W(h_s + h_r)       ← transformed prediction
```

The **true object** is `"university"`. The question every metric answers
differently is: *how close does ĥ (or Wĥ) come to `BERT("university")`
compared with every other object in the test pool?*

---

## 1. Mean Cosine Similarity

### What it is

Cosine similarity measures the **angle** between two vectors, ignoring their
magnitude. A value of 1.0 means the two vectors point in exactly the same
direction; 0.0 means they are orthogonal (unrelated).

$$
\text{mean\_cos\_sim} = \frac{1}{N} \sum_{i=1}^{N}
\cos\!\bigl(\hat{o}_i,\; h_{o_i}\bigr)
$$

### Worked example

| Triple | $cos(\hat{o}, h_o)$ |
|---|---|
| (Aarhus University, instance of, **university**) | 0.71 |
| (Don S. Davis, educated at, **Missouri State University**) | 0.58 |
| (Kōbō Abe, occupation, **photographer**) | 0.49 |
| **Average** | **0.59** |

A mean cosine similarity of 0.59 means the model's predicted vectors lean in
roughly the right direction, but are far from perfectly aligned with the true
object representations.

### What it does *not* tell you

A decent cosine similarity does not guarantee good retrieval. There may be
hundreds of other objects in the pool whose vectors are even closer to ĥ than
the true one. That is what MRR and Hits@k capture.

---

## 2. MRR — Mean Reciprocal Rank

### What it is

For each query, sort **all** test objects by their cosine similarity to $\hat{o}$ (i.e.,
$Wh_o$), find the position of the correct object in that ranked list, and take its
reciprocal. Average over all queries.

$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$

### Worked example

Suppose the test pool contains 4 242 unique objects. For three queries:

| Triple | True object | rank | 1 / rank |
|---|---|---|---|
| (Aarhus University, instance of, ?) | university | **1** | 1.000 |
| (Don S. Davis, educated at, ?) | Missouri State University | **8** | 0.125 |
| (Kōbō Abe, occupation, ?) | photographer | **3** | 0.333 |
| | | **MRR** | **0.486** |

An MRR of 0.486 means that on average the true object sits just outside the top
2 results — the model is in the right neighbourhood but not always first.

### How to read MRR values

| MRR | Interpretation |
|---|---|
| 1.00 | Perfect — always rank 1 |
| 0.50 | True object is rank 1 or 2 on average |
| 0.10 | True object is around rank 10 on average |
| < 0.05 | Model performs near random |

MRR is **heavily penalised by large ranks**. Dropping from rank 1 to rank 2
costs 0.5 points; from rank 1 to rank 10 costs 0.9 points. This makes it a
sensitive signal for whether the model consistently puts the right answer first.

---

## 3. Hits@k

### What it is

A binary, threshold-based metric: for each query, did the true object appear in
the **top-k** results? Average the binary outcome over all queries.

$$
\text{Hits@}k = \frac{1}{N} \left|\bigl\{i : \text{rank}_i \leq k\bigr\}\right|
$$

We report three thresholds: **k = 1**, **k = 3**, and **k = 10**.

### Worked example (N = 4 242 candidates)

| Triple | True object | rank | @1 | @3 | @10 |
|---|---|---|---|---|---|
| (Aarhus University, instance of, ?) | university | 1 | ✓ | ✓ | ✓ |
| (Vermont, country, ?) | United States of America | 2 | ✗ | ✓ | ✓ |
| (Kōbō Abe, occupation, ?) | photographer | 3 | ✗ | ✓ | ✓ |
| (baron, instance of, ?) | noble title | 12 | ✗ | ✗ | ✗ |
| **Hits@k** | | | **0.25** | **0.75** | **0.75** |

### How to read Hits@k values

- **Hits@1** is the strictest: the model must name the exact correct object.
  It is equivalent to accuracy in a closed-world retrieval setting.
- **Hits@3** is useful when several objects are plausible answers (e.g.
  multiple universities for "educated at").
- **Hits@10** gives credit to a model that understands the relation well enough
  to shortlist the correct answer even if it is not first.

---

## How the three metrics complement each other

```
         mean_cos_sim         MRR              Hits@1
              │                │                 │
         "Is the vector    "How high is     "Does the model
          pointing in       the true         get it exactly
          the right         object in        right?"
          direction?"       the ranking?"
              │                │                 │
           soft,            sensitive         binary,
         continuous         to rank           strict
```

In practice, look at all three together:

- **High cos\_sim + low MRR** → the model finds the right region of the
  embedding space but other wrong candidates cluster there too (e.g. all
  occupations sit close together, making "photographer" hard to distinguish
  from "actor" or "director").
- **Low cos\_sim + low MRR** → the additive assumption $h_s + h_r \approx h_o$
  does not hold; the vector space has no clear linear structure for this
  relation type.
- **MRR improves after W, Hits@1 does not** → W shifts the prediction closer
  to the right neighbourhood but not close enough to beat the top-1 competitor.

---

## Baseline reference point

With **N = 4 242** candidates and a random predictor:

| Metric | Expected random value |
|---|---|
| mean\_cos\_sim | ~0 (vectors are approximately orthogonal at random) |
| MRR | ≈ 1 / 4242 ≈ **0.0002** |
| Hits@1 | ≈ 1 / 4242 ≈ **0.024 %** |
| Hits@10 | ≈ 10 / 4242 ≈ **0.24 %** |

Any result meaningfully above these numbers indicates the model has captured
some relational structure in the BERT embedding space.
