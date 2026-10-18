# Paper to code map

Every table, figure and number in the submitted paper, with the script that produced it, the result extract it reads or writes, and the plot script if there is one. Scripts are run from the repo root as `python -m experiments.<folder>.<script>`. Raw run outputs go to `results/runs/` (not in the repo). The small extracts the tables and figures were read from are in `final_results/`.

## Main text

| Paper item | What it shows | Script | Result extract | Plot |
|---|---|---|---|---|
| Table 1 | dataset sizes | `factgrasp/data.py` loads `data/` | `data/*.csv` | |
| Table 2 | composed vector vs linear map vs MoT, 19 models | `main_results/all_models.py` running `main_results/ladder.py` per model | `final_results/ladder/ladder_<model>.json` (row `vanilla_pca768` is the composed vector, stage 2 rows are the maps), summary in `final_results/data/all_models_ladder.csv` | |
| Table 3 | the 4 configs (cosine vs cosine+InfoNCE, map vs MoT), mean of 19 | same ladder runs | `final_results/ladder/main_19_stage2_records.json` (all 4 configs per seed at each selected layer) | |
| Sec 4.1, "+0.016 from layer choice" | selected layer vs last layer | `main_results/selected_vs_last_layer.py --table` | `final_results/summaries/selected_vs_last_layer.csv` | |
| Sec 4.1, 0.355 vs 0.226 and entity-disjoint numbers | see Table 8 | | | |
| Figure 1, Figure 2 | method diagrams | drawn by hand, not code | | |
| Figure 3, 95.9%, IQR, -0.017 | transfer ratio for 342 ordered pairs | `transfer/transfer_all_pairs.py`, then `transfer/transfer_summary.py` | `final_results/data/transfer_all_pairs.csv`, `final_results/summaries/transfer_summary_*` | `plots/make_paper_figures.py` (panel a of `figure3_transfer`) |
| Sec 4.2, 8 pairs, 0.969 to 0.987 | transferred map vs a map trained for the pair | `transfer/fresh_map_comparison.py` | `final_results/data/fresh_map_comparison.csv` | |
| Sec 4.2, mismatched pairs 0.007 | deranged alignment control | `transfer/transfer_controls.py` | `results/runs/transfer_controls_all.json` | |
| Sec 4.2, 18,761 pairs, 95.8% vs 90.7% | alignment refit without test strings | `transfer/disjoint_alignment.py` | `final_results/data/disjoint_alignment.csv` | |
| Sec 4.2, Falcon-Mamba 0.433 vs 0.427 | transfer to a state space model, random composition alignment (`factgrasp/random_composition.py`) | `main_results/ladder.py`, `controls/untrained_floor.py`, `training_dynamics/checkpoint_reads.py`, `training_dynamics/collect_checkpoint_reads.py --item falcon_mamba` | `final_results/data/falcon_mamba.csv` | |
| Figure 4a, Sec 4.3 (28 checkpoints, 72%) | OLMo-2-1B checkpoints, native vs fixed Llama-3.1-8B map | `training_dynamics/checkpoint_reads.py` per checkpoint, `training_dynamics/collect_checkpoint_reads.py --item olmo_checkpoints` | `final_results/data/olmo2_checkpoints.csv`, `olmo2_checkpoint_summary.csv` | `plots/make_paper_figures.py` (`figure4_training_agreement`) |
| Figure 4b, 0.549, 0.977 | fact level agreement between 20 models | `fact_agreement/fact_agreement.py`, `fact_agreement/agreement_controls.py` | `final_results/data/fact_agreement_matrix.csv`, `agreement_seed_ceiling.csv`, `agreement_untrained.csv`, `agreement_object_frequency.csv` | `plots/make_paper_figures.py` |

## Appendix

| Paper item | What it shows | Script | Result extract | Plot |
|---|---|---|---|---|
| Table 4 | loss study on BERT-base | not in this repo, run by a co-author | | |
| Table 5 | selected vs last layer, 19 models | `main_results/selected_vs_last_layer.py --table` | `final_results/summaries/selected_vs_last_layer.csv` | |
| Table 6 | per model seed spread | ladder runs (`main_results/ladder.py`) | `final_results/ladder/main_19_stage2_records.json` | |
| Table 7, Figure 5 | independent layer selection | `layer_selection/independent_layer_selection.py --stage all` | `final_results/appendix/independent_layer_selection_*` | `plots/make_layer_selection_figure.py` |
| Table 8, upper block | trained MoT vs random targets, random encoder, majority object, character trigrams | `controls/bert_controls.py` (controls in `factgrasp/controls.py`) | `final_results/data/bert_controls.csv`, `bert_ladder.csv` | |
| Table 8, fine-tuned BERT 0.381 | SimKGC style fine tuning reference | `controls/finetuned_bert.py` | `final_results/data/finetuned_bert.csv` | |
| Table 8, lower block | split with no shared subject or object strings | `controls/entity_disjoint_split.py` | `final_results/data/entity_disjoint_split.csv` | |
| Untrained model floors | used in Figure 4a and the agreement controls | `controls/untrained_floor.py` | `final_results/data/untrained_floor.csv` | |
| Table 9, Figure 6 | alignment size and alignment family | `transfer/alignment_capacity.py` | `final_results/appendix/alignment_capacity_*` | `plots/make_alignment_capacity.py` |
| Appendix D, reverse map 64% and 80% | prediction sent back to model B's own space | `transfer/reference_map.py`, `transfer/reverse_map.py` | `final_results/data/reference_map.csv`, `reverse_map.csv` | |
| Table 10 | model index used in Figures 3 and 4 | model order set in `plots/palette.py` and `plots/make_paper_figures.py` | `figures/figure_model_index.md` (written by the plot script) | |
| Table 11 | post training change per pair | `post_training/post_training_pairs.py` | `final_results/data/post_training_pairs.csv` | |
| Table 11, architecture and size text | | copied from the model cards, not code | | |
| Table 12 | base vs post MRR and intervention cost | `post_training/post_training_pairs.py` with `post_training/steering.py` and `post_training/intervention_cost.py` | `final_results/data/post_training_pairs.csv`, `causal_arm_by_stage.csv`, `steering_base_vs_post.csv` | |
| Appendix E, 11.134 nats | steering through the alignment | `post_training/cross_model_steering.py` | `final_results/data/cross_model_steering.csv` | |
| Figure 7 | base vs post pairs (a) and OLMo training stages (b) | ladder runs on each stage checkpoint, collected by `post_training/post_training_pairs.py` | `final_results/data/post_training_pairs.csv`, `olmo3_post_training_trajectory.csv` | `plots/make_paper_figures.py` (`figure7_post_training`) |
| Figure 8, 4.1%, 0.493 to 0.514, router 0.387, ConceptNet 0.011 | relation level expertise and routing | `relation_expertise/relation_expertise.py` (ConceptNet ranks come from `main_results/conceptnet_ladder.py`) | `final_results/data/relation_expertise.csv`, `relation_routing.csv`, `relation_difficulty.csv` | `plots/make_paper_figures.py` (`figure8_relation_expertise`) |
| Table 13 | MoT vs prompting on 1,000 triples | `prompting/generation_baseline.py`, `prompting/likelihood_ranking.py`, `prompting/prompt_baselines.py`, `prompting/soft_prompt.py` | `final_results/data/prompt_baselines.csv`, `generation_baseline.csv`, `likelihood_ranking.csv` | |

## Notes

- The images uploaded for Figures 3 and 4 were re-exported in Overleaf. Figure 3 is panel (a) of `figure3_transfer` from `plots/make_paper_figures.py`; the data and drawing code are the same.
- Figures 7 and 8 label the axis "filtered MRR". That is the same quantity the paper text calls MRR.
- `olmo3_post_training_trajectory.csv` was put together from the ladder JSON of each OLMo stage checkpoint (SFT, DPO, final). No separate script writes it.
- Several scripts also compute diagnostics and pass or fail checks that are not in the paper. They are kept because the paper numbers come out of the same runs.
- `results/runs/` is where every script writes its raw JSON and per triple ranks. Later steps read these files, so run the steps in the README order.
