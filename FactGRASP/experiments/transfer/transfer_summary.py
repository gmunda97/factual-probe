# median, iqr, the -0.017 median change and family blocks from transfer_all_pairs.csv, cpu only
# run: python -m experiments.transfer.transfer_summary

from __future__ import annotations

import csv
import os
import statistics
import sys

from factgrasp import stats
from factgrasp.io_utils import RAW, FINAL, write_json, write_results_csv, utc_now, sha256
from factgrasp import paths

SRC = os.path.join(paths.ROOT, "final_results", "data", "transfer_all_pairs.csv")
CEIL = os.path.join(paths.ROOT, "final_results", "data", "fresh_map_comparison.csv")
REVERSE_CSV = os.path.join(paths.ROOT, "final_results", "data", "reverse_map.csv")


def med(v):
    return statistics.median(v) if v else float("nan")


def main() -> int:
    rows = list(csv.DictReader(open(SRC)))
    for r in rows:
        r["tau"] = float(r["tau"]); r["mrr_native"] = float(r["mrr_native"]); r["mrr_stitched"] = float(r["mrr_stitched"])
    models = sorted({r["A"] for r in rows}); fam = {}
    for r in rows:
        fam[r["A"]] = r["family_A"]; fam[r["B"]] = r["family_B"]
    native = {r["A"]: r["mrr_native"] for r in rows}
    n_pairs = len(rows)
    src_rows, results = [], []
    for m in models:
        as_in = [r["tau"] for r in rows if r["B"] == m]; as_ref = [r["tau"] for r in rows if r["A"] == m]
        src_rows.append({"model": m, "family": fam[m], "native_mrr": native[m], "n_directed_pairs_as_input": len(as_in),
                         "median_tau_as_input": med(as_in), "min_tau_as_input": min(as_in), "max_tau_as_input": max(as_in),
                         "n_directed_pairs_as_reference": len(as_ref), "median_tau_as_reference": med(as_ref), "min_tau_as_reference": min(as_ref), "max_tau_as_reference": max(as_ref),
                         "n_models": len(models)})
    rho_in = stats.spearman([native[m] for m in models], [r["median_tau_as_input"] for r in src_rows])
    rho_ref = stats.spearman([native[m] for m in models], [r["median_tau_as_reference"] for r in src_rows])
    fams = sorted(set(fam.values()))
    fam_rows = []
    for fa in fams:
        for fb in fams:
            v = [r["tau"] for r in rows if r["family_A"] == fa and r["family_B"] == fb]
            if not v:
                continue
            fam_rows.append({"reference_family": fa, "input_family": fb, "n_models_reference": sum(1 for m in models if fam[m] == fa),
                             "n_models_input": sum(1 for m in models if fam[m] == fb), "n_directed_pairs": len(v), "median_tau": med(v), "min_tau": min(v), "max_tau": max(v),
                             "block": "within" if fa == fb else "cross"})
    within = [r["tau"] for r in rows if r["family_A"] == r["family_B"]]; cross = [r["tau"] for r in rows if r["family_A"] != r["family_B"]]
    dec = [r["tau"] for r in rows if r["kind_A"] == r["kind_B"] == "decoder"]; enc = [r["tau"] for r in rows if "encoder" in (r["kind_A"], r["kind_B"])]
    by = {(r["A"], r["B"]): r for r in rows}
    dir_rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ab, ba = by[(a, b)], by[(b, a)]
            stronger = a if native[a] >= native[b] else b; weaker = b if stronger == a else a
            t_into_strong = by[(stronger, weaker)]["tau"]; t_into_weak = by[(weaker, stronger)]["tau"]
            dir_rows.append({"model_a": a, "model_b": b, "family_a": fam[a], "family_b": fam[b], "native_a": native[a], "native_b": native[b],
                             "tau_a_reads_b": ab["tau"], "tau_b_reads_a": ba["tau"], "asymmetry_abs": abs(ab["tau"] - ba["tau"]),
                             "stronger_native": stronger, "tau_stronger_reads_weaker": t_into_strong, "tau_weaker_reads_stronger": t_into_weak,
                             "stronger_reads_weaker_higher": int(t_into_strong > t_into_weak), "n_directed_pairs": 2})
    n_unordered = len(dir_rows)
    n_sw = sum(r["stronger_reads_weaker_higher"] for r in dir_rows)
    asym = [r["asymmetry_abs"] for r in dir_rows]
    ceil = list(csv.DictReader(open(CEIL))) if os.path.exists(CEIL) else []
    port = [float(r["portability"]) for r in ceil if r.get("portability")]
    w = list(csv.DictReader(open(REVERSE_CSV)))
    ret_rows = []
    for r in w:
        if r["model"] == r["hub_id"]:
            continue
        ret_rows.append({"model": r["model"], "family": fam.get(r["model"], ""), "native_own_space": float(r["native"]), "hub_reference_space": float(r["hub"]),
                         "ratio_reference_space": float(r["hub"]) / float(r["native"]), "own_space_query_reverse": float(r["hub_matched_own_pool"]),
                         "ratio_query_reverse": float(r["ratio_matched"]), "own_space_object_reverse": float(r["hub_matched_own_pool_objfit"]),
                         "ratio_object_reverse": float(r["hub_matched_own_pool_objfit"]) / float(r["native"]), "native_floor": float(r["native_floor"]), "hub_floor": float(r["hub_floor"]),
                         "hub": r["hub_id"], "n_models": 18})
    med_q = med([r["ratio_query_reverse"] for r in ret_rows]); med_o = med([r["ratio_object_reverse"] for r in ret_rows]); med_ref = med([r["ratio_reference_space"] for r in ret_rows])
    def wcsv(path, rs):
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rs[0].keys())); wr.writeheader(); [wr.writerow(x) for x in rs]
    os.makedirs(FINAL, exist_ok=True)
    wcsv(os.path.join(FINAL, "transfer_summary_source_medians.csv"), src_rows); wcsv(os.path.join(FINAL, "transfer_summary_family_blocks.csv"), fam_rows)
    wcsv(os.path.join(FINAL, "transfer_summary_directionality.csv"), dir_rows); wcsv(os.path.join(FINAL, "transfer_summary_return_space.csv"), ret_rows)
    for m in models:
        s = next(x for x in src_rows if x["model"] == m)
        results.append({"queue_id": "transfer summary", "condition_id": f"input_median::{m}", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "test", "seed": 0, "layer": "chosen",
                        "selection_role": "descriptive", "metric": "median_tau_as_input", "value": s["median_tau_as_input"], "n": 18, "status": "completed", "source_path": SRC})
        results.append({"queue_id": "transfer summary", "condition_id": f"reference_median::{m}", "model": m, "checkpoint": m, "dataset": "wikidata", "split": "test", "seed": 0, "layer": "chosen",
                        "selection_role": "descriptive", "metric": "median_tau_as_reference", "value": s["median_tau_as_reference"], "n": 18, "status": "completed", "source_path": SRC})
    for r in ret_rows:
        for k in ("ratio_reference_space", "ratio_query_reverse", "ratio_object_reverse"):
            results.append({"queue_id": "transfer summary", "condition_id": f"{k}::{r['model']}", "model": r["model"], "checkpoint": r["model"], "dataset": "wikidata", "split": "test", "seed": 0, "layer": "chosen",
                            "selection_role": "descriptive", "metric": k, "value": r[k], "n": 1, "status": "completed", "source_path": REVERSE_CSV})
    for k, v in (("median_tau_all", med([r["tau"] for r in rows])), ("median_tau_within_family", med(within)), ("median_tau_cross_family", med(cross)),
                 ("median_tau_decoder_decoder", med(dec)), ("median_tau_encoder_involved", med(enc)), ("spearman_native_vs_median_tau_as_input", rho_in),
                 ("spearman_native_vs_median_tau_as_reference", rho_ref), ("share_unordered_pairs_stronger_reads_weaker_higher", n_sw / n_unordered),
                 ("median_abs_asymmetry", med(asym)), ("median_ratio_reference_space_reverse_map", med_ref), ("median_ratio_query_reverse_reverse_map", med_q), ("median_ratio_object_reverse_reverse_map", med_o),
                 ("median_portability_fresh_probe_fresh_map", med(port))):
        results.append({"queue_id": "transfer summary", "condition_id": k, "model": "all", "checkpoint": "", "dataset": "wikidata", "split": "test", "seed": 0, "layer": "chosen", "selection_role": "descriptive",
                        "metric": k, "value": v, "n": {"median_tau_all": n_pairs, "median_tau_within_family": len(within), "median_tau_cross_family": len(cross), "median_tau_decoder_decoder": len(dec),
                                                         "median_tau_encoder_involved": len(enc), "median_ratio_reference_space_reverse_map": len(ret_rows), "median_ratio_query_reverse_reverse_map": len(ret_rows),
                                                         "median_ratio_object_reverse_reverse_map": len(ret_rows), "median_portability_fresh_probe_fresh_map": len(port)}.get(k, n_unordered if "unordered" in k or "asymmetry" in k else len(models)),
                        "status": "completed", "source_path": SRC if "reverse_map" not in k and "fresh_map" not in k else (REVERSE_CSV if "reverse_map" in k else CEIL)})
    write_results_csv(os.path.join(FINAL, "transfer_summary_results.csv"), results)
    summary = {"status": "completed", "conditions_planned": len(results), "conditions_completed": len(results), "conditions_failed": 0,
               "primary_results": {"n_models": len(models), "n_directed_pairs": n_pairs, "median_tau_all": med([r["tau"] for r in rows]),
                                   "median_tau_within_family": med(within), "n_within": len(within), "median_tau_cross_family": med(cross), "n_cross": len(cross),
                                   "median_tau_decoder_decoder": med(dec), "n_decoder_decoder": len(dec), "median_tau_encoder_involved": med(enc), "n_encoder_involved": len(enc),
                                   "spearman_native_vs_median_tau_as_input": rho_in, "spearman_native_vs_median_tau_as_reference": rho_ref,
                                   "input_medians_lowest": sorted(((r["median_tau_as_input"], r["model"]) for r in src_rows))[:3],
                                   "input_medians_highest": sorted(((r["median_tau_as_input"], r["model"]) for r in src_rows))[-3:],
                                   "directionality": {"n_unordered_pairs": n_unordered, "stronger_reads_weaker_higher": n_sw, "median_abs_asymmetry": med(asym), "max_abs_asymmetry": max(asym)},
                                   "return_space_reverse_map": {"n_models": len(ret_rows), "median_ratio_reference_space": med_ref, "median_ratio_query_reverse": med_q, "median_ratio_object_reverse": med_o},
                                   "fresh_probe_ceiling_fresh_map": {"n_pairs": len(port), "median_portability": med(port), "min": min(port) if port else None, "max": max(port) if port else None}},
               "null_or_adverse_results": {"weakest_inputs": [f"{m}: median tau as input {v:.3f}" for v, m in sorted(((r['median_tau_as_input'], r['model']) for r in src_rows))[:2]],
                                           "own_space_return": f"the reference-space ratio {med_ref:.3f} falls to {med_q:.3f} (query-fitted reverse map) and {med_o:.3f} (object-fitted) in the input model's own candidate space"},
               "protocol_deviations": [], "limitations": ["342 directed pairs from 19 models are correlated observations; no pair is an independent replication",
                                                           "the matrix uses the historical last-token ModernBERT-base cache (main table: mean pooling)",
                                                           "the Spearman between native MRR and median tau as input is descriptive over 19 models", "seed-0 stitches"],
               "artifact_paths": [os.path.join(FINAL, f) for f in ("transfer_summary_source_medians.csv", "transfer_summary_family_blocks.csv", "transfer_summary_directionality.csv", "transfer_summary_return_space.csv", "transfer_summary_results.csv", "transfer_summary_summary.json", "transfer_summary_report.md")],
               "inputs_sha256": {SRC: sha256(SRC), CEIL: sha256(CEIL) if os.path.exists(CEIL) else None, REVERSE_CSV: sha256(REVERSE_CSV)}, "completed_at_utc": utc_now()}
    write_json(os.path.join(FINAL, "transfer_summary_summary.json"), summary); write_json(os.path.join(RAW, "transfer_summary_summary.json"), summary)
    print(json.dumps(summary["primary_results"], indent=1, default=str)); print(summary["null_or_adverse_results"])
    return 0


if __name__ == "__main__":
    import json
    sys.exit(main())
