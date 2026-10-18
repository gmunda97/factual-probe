# collects the reads into final_results/data/olmo2_checkpoints.csv olmo2_checkpoint_summary.csv falcon_mamba.csv
# the 72 percent at 21b tokens is (native at 21b minus floor) / (released model minus floor)
# run: python -m experiments.training_dynamics.collect_checkpoint_reads --item olmo_checkpoints   (or --item falcon_mamba)

from __future__ import annotations

import argparse
import json
import os
import sys

from factgrasp import encoders as E, paths, registry

NAME = "collect_checkpoint_reads"
RUNS = os.path.join(paths.ROOT, "results", "runs")
HUB, HUB2 = "meta-llama/Llama-3.1-8B", "Qwen/Qwen3.5-9B"
THRESHOLDS = {
    "checkpoint_transfer": "the universal readout tracks the native readout across pretraining: hub-probe MRR at least 0.9 of the "
           "native probe at every checkpoint where the native probe is at least 0.05 above the floor (green); at 6 of 8 "
           "(amber); fewer (red, the universality claim restricted to trained models)",
    "checkpoint_rise": "relational readability rises with training and most of it arrives early: native MRR at the checkpoint "
           "nearest one fifth of training at least half of the final value (green); one third to one half (amber); "
           "below one third (red)",
    "mamba_objective": "the objective result holds without attention: InfoNCE beats cosine with the interval excluding 0, and the "
           "mixture's gain is smaller under InfoNCE (green); one of the two (amber); neither (red)",
    "mamba_transfer": "the universal readout reads a model with no attention: hub-probe MRR at least 0.85 of native through the "
           "zero-label stitch (green); 0.7 to 0.85 (amber); below 0.7 (red)",
}


def reads(tag: str) -> list:
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, f"checkpoint_reads_{tag}__*.json"))):
        out.extend(v for v in json.load(open(p)).get("rows", {}).values() if isinstance(v, dict) and "hubs" in v)
    return out


def f6(v):
    return "" if v is None else f"{v:.6f}"


def collect_falcon_mamba(csv_dir: str) -> int:
    run = registry.Run(NAME, "falcon_mamba", config={}, thresholds={k: v for k, v in THRESHOLDS.items() if k.startswith("falcon mamba")}, out_dir=RUNS)
    model = "tiiuae/falcon-mamba-7b"
    blob = json.load(open(os.path.join(paths.RUNS, f"ladder_{E.model_slug(model)}.json")))
    lad = blob["summary"]
    r = reads("falcon_mamba")[0]
    gi = lad.get("gap_infonce_vs_cosine", {}); gm = lad.get("gap_mixture_vs_single", {})
    mix_inf, single_inf = lad.get("headline_MRR_mixture_infonce"), lad.get("MRR_single_infonce")
    by_row = {}
    for k, v in (blob.get("rows") or {}).items():
        if not k.startswith("{") or not isinstance(v, dict) or "MRR" not in v:
            continue
        cfg = json.loads(k)
        if cfg.get("stage") == 2 and cfg.get("layer") == lad.get("best_layer"):
            by_row.setdefault(v.get("row") or cfg.get("row"), []).append(float(v["MRR"]))
    row_mrr = lambda name: (sum(by_row[name]) / len(by_row[name])) if by_row.get(name) else None
    mc, sc, mi, si = row_mrr("mixture8_cosine"), row_mrr("single_cosine"), row_mrr("mixture8_infonce"), row_mrr("single_infonce")
    gain_cos = (mc - sc) if None not in (mc, sc) else None
    gain_inf = (mi - si) if None not in (mi, si) else None
    row_means = {"single_cosine": sc, "mixture8_cosine": mc, "single_infonce": si, "mixture8_infonce": mi}
    cond1 = bool(gi.get("significant")) and float(gi.get("ci_low", 0)) > 0
    cond2 = (gain_inf is not None and gain_cos is not None and gain_inf < gain_cos)
    objective_v = "green" if (cond1 and cond2) else "amber" if (cond1 or cond2) else "red"
    ratio = r["hubs"][HUB]["ratio_zero_label"]
    mamba_v = "green" if ratio >= 0.85 else "amber" if ratio >= 0.7 else "red"
    run.set_summary({"verdicts": {"mamba_objective": objective_v, "mamba_transfer": mamba_v},
                     "measured": {"mamba_objective_gap_infonce_minus_cosine": gi, "mamba_objective_mixture_gain_under_cosine": gain_cos, "mamba_objective_mixture_gain_under_infonce": gain_inf,
                                  "mamba_objective_conditions": {"infonce_beats_cosine_interval_excludes_0": cond1, "mixture_gain_smaller_under_infonce": cond2},
                                  "mamba_transfer_ratio_zero_label_llama_hub": round(ratio, 4), "mamba_transfer_hub_MRR_zero_label": round(r["hubs"][HUB]["hub_MRR_zero_label"], 4),
                                  "mamba_transfer_native_MRR": round(r["native_MRR"], 4), "mamba_transfer_second_hub_ratio_zero_label": round(r["hubs"][HUB2]["ratio_zero_label"], 4),
                                  "hub_query_stitch_ratio": {h: round(v["ratio"], 4) for h, v in r["hubs"].items()},
                                  "matched_ratio": {h: round(v["ratio_matched"], 4) for h, v in r["hubs"].items()},
                                  "gain_above_floor": {h: {"hub": v["hub_gain_above_floor"], "native": v["native_gain_above_floor"]} for h, v in r["hubs"].items()},
                                  "ladder": {k: lad.get(k) for k in ("best_layer", "n_layers", "pooling", "headline_MRR_mixture_infonce", "MRR_single_infonce")},
                                  "ladder_stage2_row_means": row_means}})
    run.flush()
    with open(os.path.join(csv_dir, "falcon_mamba.csv"), "w") as fh:
        fh.write("model,family,size_b,base_or_post,layers,best_layer,relative_depth,pooling,MRR_mixture_infonce,MRR_single_infonce,"
                 "gap_infonce_minus_cosine,gap_infonce_significant,gap_mixture_minus_single,gap_mixture_significant,mixture_gain_under_cosine,"
                 "mixture_gain_under_infonce,native_MRR_reads,floor_MRR,hub,hub_MRR,ratio,hub_MRR_zero_label,ratio_zero_label,hub_matched_own_pool,"
                 "ratio_matched,hub_floor_MRR,hub_gain_above_floor,native_gain_above_floor\n")
        for h, v in r["hubs"].items():
            fh.write(f"{model},FalconMamba,7.27,base,{lad.get('n_layers')},{lad.get('best_layer')},{lad.get('lambda_star') if lad.get('lambda_star') is not None else (lad.get('best_layer') or 0) / (lad.get('n_layers') or 1):.6f},"
                     f"{lad.get('pooling')},{f6(mix_inf)},{f6(single_inf)},{f6(gi.get('diff'))},{gi.get('significant')},{f6(gm.get('diff'))},{gm.get('significant')},"
                     f"{f6(gain_cos)},{f6(gain_inf)},{f6(r['native_MRR'])},{f6(r['native_floor'])},{h},{f6(v['hub_MRR'])},{f6(v['ratio'])},"
                     f"{f6(v['hub_MRR_zero_label'])},{f6(v['ratio_zero_label'])},{f6(v['hub_matched_own_pool'])},{f6(v['ratio_matched'])},"
                     f"{f6(v['hub_floor'])},{f6(v['hub_gain_above_floor'])},{f6(v['native_gain_above_floor'])}\n")
    print(f"=== SUMMARY {NAME} u2 ===", flush=True)
    print(f"mamba_objective  {objective_v}   InfoNCE minus cosine {gi.get('diff')} [{gi.get('ci_low')}, {gi.get('ci_high')}]; mixture gain under cosine {gain_cos}, under InfoNCE {gain_inf}", flush=True)
    print(f"mamba_transfer  {mamba_v}   zero-label stitch into the Llama hub {r['hubs'][HUB]['hub_MRR_zero_label']:.4f} of native {r['native_MRR']:.4f} (ratio {ratio:.3f}); "
          f"Qwen hub ratio {r['hubs'][HUB2]['ratio_zero_label']:.3f}; query stitch ratio {r['hubs'][HUB]['ratio']:.3f}; matched {r['hubs'][HUB]['ratio_matched']:.3f}; "
          f"floors: native {r['native_floor']}, hub {r['hubs'][HUB]['hub_floor']}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def collect_olmo_coarse(csv_dir: str, stage1_end: float = 4001.0) -> int:
    run = registry.Run(NAME, "olmo_coarse", config={}, thresholds={k: v for k, v in THRESHOLDS.items() if k.startswith("olmo checkpoints")}, out_dir=RUNS)
    rows = reads("olmo2_coarse")
    main = next((r for r in rows if r["revision"] is None), None)
    ck = sorted([r for r in rows if r["revision"] is not None], key=lambda r: r["tokens_B"] or 0)
    floor = main["native_floor"] if main else None
    above = [r for r in ck if floor is not None and r["native_MRR"] - floor >= 0.05]
    ok = [r for r in above if r["hubs"][HUB]["ratio_zero_label"] >= 0.9]
    n_stage1 = len(ck)
    if above and len(ok) == len(above):
        transfer_v = "green"
    elif len(ok) >= 6:
        transfer_v = "amber"
    else:
        transfer_v = "red"
    target = stage1_end / 5.0
    near = min(ck, key=lambda r: abs((r["tokens_B"] or 0) - target)) if ck else None
    frac = (near["native_MRR"] / main["native_MRR"]) if (near and main) else None
    rise_v = "red" if frac is None else "green" if frac >= 0.5 else "amber" if frac >= 1 / 3 else "red"
    run.set_summary({"verdicts": {"checkpoint_transfer": transfer_v, "checkpoint_rise": rise_v},
                     "measured": {"checkpoint_transfer_checkpoints_stage1": n_stage1, "checkpoint_transfer_native_at_least_0.05_above_floor": len(above),
                                  "checkpoint_transfer_of_those_hub_at_least_0.9": len(ok), "checkpoint_transfer_floor_MRR": floor,
                                  "checkpoint_transfer_ratio_zero_label_by_checkpoint": {(r["revision"] or "main"): round(r["hubs"][HUB]["ratio_zero_label"], 4) for r in ck + ([main] if main else [])},
                                  "checkpoint_rise_target_tokens_B": target, "checkpoint_rise_nearest_checkpoint": near["revision"] if near else None,
                                  "checkpoint_rise_native_at_nearest": round(near["native_MRR"], 4) if near else None,
                                  "checkpoint_rise_native_final_main": round(main["native_MRR"], 4) if main else None,
                                  "checkpoint_rise_fraction": round(frac, 4) if frac is not None else None,
                                  "native_by_checkpoint": {(r["revision"] or "main"): round(r["native_MRR"], 4) for r in ck + ([main] if main else [])},
                                  "hub_query_stitch_ratio_by_checkpoint": {(r["revision"] or "main"): round(r["hubs"][HUB]["ratio"], 4) for r in ck + ([main] if main else [])}}})
    run.flush()
    with open(os.path.join(csv_dir, "olmo2_checkpoint_summary.csv"), "w") as fh:
        fh.write("revision,tokens_B,layer,seed,native_MRR,native_MRR_by_seed,floor_MRR,hub,hub_MRR,ratio,hub_MRR_zero_label,ratio_zero_label,"
                 "hub_matched_own_pool,ratio_matched,hub_floor_MRR,hub_gain_above_floor,native_gain_above_floor,above_floor_by_0.05\n")
        for r in ck + ([main] if main else []):
            for h, v in r["hubs"].items():
                by = ";".join(f"{s['MRR']:.4f}" for s in r["native_per_seed"])
                fh.write(f"{r['revision'] or 'main'},{r['tokens_B'] if r['tokens_B'] is not None else ''},{r['layer']},mean,{f6(r['native_MRR'])},{by},"
                         f"{f6(r['native_floor'])},{h},{f6(v['hub_MRR'])},{f6(v['ratio'])},{f6(v['hub_MRR_zero_label'])},{f6(v['ratio_zero_label'])},"
                         f"{f6(v['hub_matched_own_pool'])},{f6(v['ratio_matched'])},{f6(v['hub_floor'])},{f6(v['hub_gain_above_floor'])},"
                         f"{f6(v['native_gain_above_floor'])},{int(floor is not None and r['native_MRR'] - floor >= 0.05)}\n")
    print(f"=== SUMMARY {NAME} u1 ===", flush=True)
    print(f"checkpoint_transfer  {transfer_v}   {len(above)} of {n_stage1} stage-1 checkpoints at least 0.05 above the floor ({floor}); hub (zero-label) at least 0.9 of native on {len(ok)} of those", flush=True)
    print(f"checkpoint_rise  {rise_v}   native at {near['revision'] if near else None} ({near['tokens_B'] if near else None} B, nearest {target:.0f} B): "
          f"{near['native_MRR'] if near else float('nan'):.4f} of final {main['native_MRR'] if main else float('nan'):.4f} = {frac if frac is not None else float('nan'):.3f}", flush=True)
    for r in ck + ([main] if main else []):
        a = r["hubs"][HUB]
        print(f"  {(r['revision'] or 'main'):40s} {str(r['tokens_B']):>6s} B native {r['native_MRR']:.4f} floor {r['native_floor']:.4f} | hub {a['hub_MRR']:.4f} ({a['ratio']:.3f}) "
              f"zero-label {a['hub_MRR_zero_label']:.4f} ({a['ratio_zero_label']:.3f}) matched {a['hub_matched_own_pool']:.4f} ({a['ratio_matched']:.3f}) "
              f"hub floor {(a['hub_floor'] if a['hub_floor'] is not None else float('nan')):.4f}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def collect_olmo_checkpoints(csv_dir: str, stage1_end: float = 4001.0) -> int:
    thr = {"checkpoint_dense": "the universal readout tracks the native readout throughout the rise: hub-probe ratio (zero-label stitch, hub's "
                  "space) at least 0.9 at every checkpoint whose native probe is at least 0.05 above the floor (green); at three "
                  "quarters of them (amber); fewer (red)",
           "checkpoint_floor": "relational readability arrives early: the native probe reaches half of its final gain above the floor by 100 B "
                  "tokens (green); between 100 and 300 B (amber); later (red, reported as it is)"}
    run = registry.Run(NAME, "olmo_checkpoints", config={}, thresholds=thr, out_dir=RUNS)
    early = reads("olmo2_checkpoints")
    coarse_rows = reads("olmo2_coarse")
    main = next((r for r in coarse_rows if r["revision"] is None), None)
    allrows = sorted([r for r in early + coarse_rows if r["revision"] is not None], key=lambda r: r["tokens_B"] or 0)
    floor = main["native_floor"]
    final = main["native_MRR"]
    half = floor + 0.5 * (final - floor)
    above = [r for r in allrows if r["native_MRR"] - floor >= 0.05]
    ok = [r for r in above if r["hubs"][HUB]["ratio_zero_label"] >= 0.9]
    early_above = [r for r in early if r["native_MRR"] - floor >= 0.05]
    early_ok = [r for r in early_above if r["hubs"][HUB]["ratio_zero_label"] >= 0.9]
    frac_ok = (len(ok) / len(above)) if above else 0.0
    dense_v = "green" if (above and len(ok) == len(above)) else "amber" if frac_ok >= 0.75 else "red"
    pts = [(r["tokens_B"], r["native_MRR"]) for r in allrows] + [(main["tokens_B"], final)]
    t_half, seg = None, None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if v0 < half <= v1:
            t_half = t0 + (half - v0) * (t1 - t0) / (v1 - v0) if v1 != v0 else t1
            seg = (t0, t1); break
        if v0 >= half and t0 == pts[0][0]:
            t_half, seg = t0, (t0, t0); break
    floor_v = "red" if t_half is None else "green" if t_half <= 100 else "amber" if t_half <= 300 else "red"
    run.set_summary({"verdicts": {"checkpoint_dense": dense_v, "checkpoint_floor": floor_v},
                     "measured": {"checkpoint_dense_checkpoints_total_stage1": len(allrows), "checkpoint_dense_early_checkpoints": len(early),
                                  "checkpoint_dense_above_floor_by_0.05": len(above), "checkpoint_dense_of_those_hub_at_least_0.9": len(ok),
                                  "checkpoint_dense_early_above_floor": len(early_above), "checkpoint_dense_early_of_those_hub_at_least_0.9": len(early_ok),
                                  "checkpoint_dense_ratio_zero_label_by_tokens": {str(r["tokens_B"]): round(r["hubs"][HUB]["ratio_zero_label"], 4) for r in allrows},
                                  "checkpoint_floor_floor": floor, "checkpoint_floor_final_main": final, "checkpoint_floor_half_gain_level": round(half, 4),
                                  "checkpoint_floor_tokens_B_at_half_gain": (round(t_half, 1) if t_half is not None else None), "checkpoint_floor_segment_B": seg,
                                  "native_by_tokens": {str(r["tokens_B"]): round(r["native_MRR"], 4) for r in allrows}}})
    run.flush()
    with open(os.path.join(csv_dir, "olmo2_checkpoints.csv"), "w") as fh:
        fh.write("revision,tokens_B,layer,seed,native_MRR,native_MRR_by_seed,floor_MRR,hub,hub_MRR,ratio,hub_MRR_zero_label,ratio_zero_label,"
                 "hub_matched_own_pool,ratio_matched,hub_floor_MRR,hub_gain_above_floor,native_gain_above_floor,above_floor_by_0.05,set\n")
        for r in allrows + [main]:
            for h, v in r["hubs"].items():
                by = ";".join(f"{s['MRR']:.4f}" for s in r["native_per_seed"])
                fh.write(f"{r['revision'] or 'main'},{r['tokens_B'] if r['tokens_B'] is not None else ''},{r['layer']},mean,{f6(r['native_MRR'])},{by},"
                         f"{f6(r['native_floor'])},{h},{f6(v['hub_MRR'])},{f6(v['ratio'])},{f6(v['hub_MRR_zero_label'])},{f6(v['ratio_zero_label'])},"
                         f"{f6(v['hub_matched_own_pool'])},{f6(v['ratio_matched'])},{f6(v['hub_floor'])},{f6(v['hub_gain_above_floor'])},"
                         f"{f6(v['native_gain_above_floor'])},{int(r['native_MRR'] - floor >= 0.05)},{'checkpoint_dense' if r in early else 'olmo checkpoints'}\n")
    print(f"=== SUMMARY {NAME} u1c ===", flush=True)
    print(f"checkpoint_dense  {dense_v}   {len(above)} of {len(allrows)} stage-1 checkpoints at least 0.05 above the floor ({floor:.4f}); hub (zero-label) at least 0.9 of native on "
          f"{len(ok)} of those ({len(early_ok)} of {len(early_above)} among the nineteen early ones)", flush=True)
    print(f"checkpoint_floor  {floor_v}   half of the final gain above the floor is {half:.4f} (floor {floor:.4f}, final {final:.4f}); first reached at "
          f"{t_half if t_half is not None else float('nan'):.1f} B tokens by interpolation between {seg}", flush=True)
    for r in allrows + [main]:
        a = r["hubs"][HUB]
        print(f"  {(r['revision'] or 'main'):40s} {str(r['tokens_B']):>6s} B native {r['native_MRR']:.4f} (+{r['native_MRR'] - floor:+.3f}) | hub {a['hub_MRR']:.4f} ({a['ratio']:.3f}) "
              f"zero-label {a['hub_MRR_zero_label']:.4f} ({a['ratio_zero_label']:.3f}) matched {a['hub_matched_own_pool']:.4f} ({a['ratio_matched']:.3f})", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", choices=["olmo_coarse", "falcon_mamba", "olmo_checkpoints"], required=True)
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    args = ap.parse_args()
    os.makedirs(args.csv_dir, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
    return {"falcon_mamba": collect_falcon_mamba, "olmo_coarse": collect_olmo_coarse, "olmo_checkpoints": collect_olmo_checkpoints}[args.item](args.csv_dir)


if __name__ == "__main__":
    sys.exit(main())
