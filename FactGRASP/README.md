# FactGRASP

Code for **FactGRASP: Graph-Aware Relational Activation Shaping for Factual Knowledge Discovery, Recall, and Transfer in LLMs**.

A frozen language model encodes the subject, the relation and the object of a knowledge graph triple separately. FactGRASP composes x = h_s + h_r and learns a linear map, or a relation-gated mixture of eight maps (MoT), that sends x to the object vector. Candidates are ranked by cosine similarity, and the map is evaluated with filtered MRR and Hits@k. A map trained on one model can then be reused on another model after a ridge alignment of their composed vectors.

`RESULTS_MAP.md` lists every table and figure of the paper with the script and the result file behind it.

## Layout

```
factgrasp/        shared helpers: data loading, embedding and caching, the maps and losses,
                  filtered ranking metrics, cross model alignment, control baselines, io
experiments/      one folder per part of the paper, each script is one experiment
  main_results/       Tables 2, 3, 5, 6 (the 19 model ladder) and the ConceptNet ladder
  layer_selection/    Table 7, Figure 5
  controls/           Table 8, untrained model floors
  transfer/           Figure 3, the 8 pair comparison, Table 9, Figure 6, reverse map
  training_dynamics/  Figure 4a, OLMo-2 checkpoints, Falcon-Mamba
  fact_agreement/     Figure 4b
  post_training/      Tables 11, 12, Figure 7
  relation_expertise/ Figure 8
  prompting/          Table 13
plots/            scripts that draw the paper figures from final_results/
data/             Wikidata5M subset and ConceptNet subset, train / valid / test
final_results/    the small result extracts the paper tables and figures were read from
```

Running an experiment creates `cache/` (hidden states), `results/` (raw run JSON) and `logs/`. None of these are in the repo.

## Setup

```bash
conda create -n factgrasp python=3.11
conda activate factgrasp
pip install -r requirements.txt
```

Gated models (Llama, Gemma) need `huggingface-cli login`. Run everything from the repo root.

## Data

`data/wikidata5m_42k_{train,valid,test}_relations.csv` hold 33,937 / 4,242 / 4,243 triples over 191 relations, a subset of Wikidata5M. `data/conceptnet_en_{train,valid,test}_relations.csv` hold 29,081 / 3,635 / 3,635 triples over 9 relations. The columns are `subject, object, similarity, pred_value`, where `pred_value` is the relation. Test queries are ranked against the 2,766 unique object strings of the test split. Filtering removes every other known correct object from the train, valid and test splits.

### How the dataset was built

<!-- to be filled in -->

## Reproducing the paper

The figures can be redrawn from the shipped extracts without a GPU:

```bash
python plots/make_paper_figures.py           # Figures 3, 4, 7, 8
python plots/make_alignment_capacity.py      # Figure 6, Table 9
python plots/make_layer_selection_figure.py  # Figure 5, Table 7
```

To rerun the experiments themselves, the order matters, because later steps read the maps and caches written by earlier ones.

1. **Main ladder (Tables 2, 3, 5, 6).** This runs once per model. It embeds the strings, sweeps layers, picks the layer with the best validation MRR, and trains the linear map and the MoT under both losses with 3 seeds.
   ```bash
   python -m experiments.main_results.all_models --model meta-llama/Llama-3.1-8B
   python -m experiments.main_results.selected_vs_last_layer --table
   python -m experiments.main_results.conceptnet_ladder --model meta-llama/Llama-3.1-8B   # ConceptNet ranks for Figure 8
   ```
2. **Controls (Table 8).**
   ```bash
   python -m experiments.controls.bert_controls
   python -m experiments.controls.untrained_floor --models meta-llama/Llama-3.1-8B
   python -m experiments.controls.entity_disjoint_split
   python -m experiments.controls.finetuned_bert
   ```
3. **Transfer (Figure 3, Section 4.2, Appendix D).** This needs step 1 for all 19 models.
   ```bash
   python -m experiments.transfer.transfer_all_pairs
   python -m experiments.transfer.transfer_controls
   python -m experiments.transfer.fresh_map_comparison
   python -m experiments.transfer.transfer_summary
   python -m experiments.transfer.disjoint_alignment
   python -m experiments.transfer.alignment_capacity
   python -m experiments.transfer.reference_map
   python -m experiments.transfer.reverse_map
   ```
4. **Training dynamics and fact agreement (Figure 4).** Run `experiments/training_dynamics/checkpoint_reads.py` once per OLMo-2-1B checkpoint; the exact command is at the top of the script. Then collect the checkpoint and Falcon-Mamba runs and compute the fact agreement:
   ```bash
   python -m experiments.training_dynamics.collect_checkpoint_reads --item olmo_checkpoints
   python -m experiments.training_dynamics.collect_checkpoint_reads --item falcon_mamba
   python -m experiments.fact_agreement.fact_agreement
   python -m experiments.fact_agreement.agreement_controls
   ```
5. **Appendix E.** Post-training (`experiments/post_training/`), relation expertise (`experiments/relation_expertise/`) and prompting (`experiments/prompting/`).

Every script has a short header that says which paper result it produces and how to run it. Most scripts take `--smoke` for a quick check on a few rows.

The PCA step uses `torch.pca_lowrank`, which is randomized, so a rerun can move scores in the third decimal.

## Compute

All language models stay frozen; only the maps and alignments are trained. Embedding a model needs one GPU. The 7B to 12B models were run in bf16 on a single 48 GB card. Training the maps, the transfer analysis and the plots are cheap once the hidden states are cached. On a shared machine `factgrasp/gpu_guard.py` picks a free card; set `FACTGRASP_FORCE_GPU=<index>` to choose one yourself.

## License and citation

To be added.
