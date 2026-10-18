# tables 11 and 12 and figure 7a: the 7 base vs post trained pairs, MoT mrr from the ladder runs and the
# cost columns from steering and intervention_cost. the architecture and size text of table 11 is from the model cards
# figure 7b (olmo stages) uses the ladder runs on the sft and dpo checkpoints, final_results/data/olmo3_post_training_trajectory.csv

from __future__ import annotations

import argparse
import json
import os
import sys

from factgrasp import encoders as E, paths, registry

NAME = "post_training_pairs"
RUNS = os.path.join(paths.ROOT, "results", "runs")

PAIRS = [
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Olmo-3-7B", "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct"),
    ("Gemma-4-E4B", "google/gemma-4-E4B", "google/gemma-4-E4B-it"),
    ("Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B"),
    ("Gemma-4-E2B", "google/gemma-4-E2B", "google/gemma-4-E2B-it"),
    ("Gemma-4-12B", "google/gemma-4-12B", "google/gemma-4-12B-it"),
    ("Olmo-Hybrid-7B", "allenai/Olmo-Hybrid-7B", "allenai/Olmo-Hybrid-Instruct-DPO-7B"),
]


def _load(path: str):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def measurements(model: str) -> dict:
    slug = E.model_slug(model)
    out = {"model": model, "readability": None, "steer": None, "steer_ci_low": None,
           "best_alpha": None, "ppl_change": None, "missing": []}
    lad = _load(os.path.join(paths.RUNS, f"ladder_{slug}.json"))
    if lad and (lad.get("summary") or {}).get("headline_MRR_mixture_infonce") is not None:
        out["readability"] = lad["summary"]["headline_MRR_mixture_infonce"]
        out["layer"] = lad["summary"].get("best_layer")
    else:
        out["missing"].append("ladder")
    cau = _load(os.path.join(paths.RUNS, f"steering_{slug}.json"))
    if cau and (cau.get("summary") or {}).get("gain_nats_vs_random_best") is not None:
        out["steer"] = cau["summary"]["gain_nats_vs_random_best"]
        out["steer_ci_low"] = cau["summary"].get("gain_ci_low_best")
        out["best_alpha"] = cau["summary"].get("best_alpha")
    else:
        out["missing"].append("steering")
    rep = _load(os.path.join(paths.ROOT, "results", "runs",
                             f"intervention_cost_{slug}.json"))
    cap = ((rep or {}).get("summary") or {}).get("capability") or {}
    if cap.get("relative_change") is not None:
        out["ppl_change"] = cap["relative_change"]
        out["ppl_base"] = cap.get("perplexity_base")
    else:
        out["missing"].append("injection cost")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(paths.ROOT, "final_results", "data"))
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    run = registry.Run(NAME, args.tag, config=vars(args), thresholds={}, out_dir=RUNS)

    rows, complete = [], []
    for name, base, post in PAIRS:
        b, p = measurements(base), measurements(post)
        r = {"pair": name, "base_model": base, "post_model": post}
        for k in ("readability", "steer", "steer_ci_low", "ppl_change", "best_alpha"):
            r[f"{k}_base"], r[f"{k}_post"] = b.get(k), p.get(k)
        for k in ("readability", "steer", "ppl_change"):
            r[f"{k}_delta"] = (None if b.get(k) is None or p.get(k) is None
                               else p[k] - b[k])
        r["missing"] = sorted(set(b["missing"]) | set(p["missing"]))
        r["complete"] = not r["missing"]
        rows.append(r)
        run.put(registry.config_key({"pair": name}), r)
        if r["complete"]:
            complete.append(r)
    run.flush()

    def direction(key, sign):
        got = [r for r in complete if r[f"{key}_delta"] is not None]
        n = sum(1 for r in got if sign * r[f"{key}_delta"] > 0)
        return n, len(got), [r["pair"] for r in got if sign * r[f"{key}_delta"] <= 0]

    read_n, read_of, read_x = direction("readability", -1)
    steer_n, steer_of, steer_x = direction("steer", -1)
    ppl_n, ppl_of, ppl_x = direction("ppl_change", +1)

    os.makedirs(args.csv_dir, exist_ok=True)
    cols = ["pair", "base_model", "post_model", "readability_base", "readability_post",
            "readability_delta", "steer_base", "steer_post", "steer_delta",
            "steer_ci_low_base", "steer_ci_low_post", "ppl_change_base", "ppl_change_post",
            "ppl_change_delta", "complete", "missing"]
    with open(os.path.join(args.csv_dir, "post_training_pairs.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                "" if r.get(c) is None else
                (f"{r[c]:.6f}" if isinstance(r[c], float) else
                 ("\"" + ";".join(r[c]) + "\"" if isinstance(r[c], list) else str(r[c])))
                for c in cols) + "\n")

    run.set_summary({"pairs": [r["pair"] for r in rows],
                     "complete_pairs": [r["pair"] for r in complete],
                     "measured": {
                         "readability_falls": f"{read_n} of {read_of}",
                         "steering_falls": f"{steer_n} of {steer_of}",
                         "injection_cost_rises": f"{ppl_n} of {ppl_of}",
                         "exceptions": {"readability": read_x, "steering": steer_x,
                                        "injection_cost": ppl_x}},
                     "incomplete": {r["pair"]: r["missing"] for r in rows
                                    if not r["complete"]},
                     "note": "exploratory synthesis; no threshold, no verdict changes"})
    run.flush()
    print(f"=== SUMMARY {NAME} {args.tag} ===", flush=True)
    print(f"{len(complete)} complete pair(s) of {len(rows)}", flush=True)
    for r in rows:
        if not r["complete"]:
            print(f"     {r['pair']:16s} incomplete: {', '.join(r['missing'])}", flush=True)
            continue
        print(f"     {r['pair']:16s} readability {r['readability_base']:.4f} -> "
              f"{r['readability_post']:.4f}  steering {r['steer_base']:.4f} -> "
              f"{r['steer_post']:.4f} nats  injection cost "
              f"{100 * r['ppl_change_base']:+.1f}% -> {100 * r['ppl_change_post']:+.1f}%",
              flush=True)
    print(f"readability falls on {read_n} of {read_of}"
          f"{'' if not read_x else ', exceptions ' + ', '.join(read_x)}", flush=True)
    print(f"steering falls on {steer_n} of {steer_of}"
          f"{'' if not steer_x else ', exceptions ' + ', '.join(steer_x)}", flush=True)
    print(f"injection cost rises on {ppl_n} of {ppl_of}"
          f"{'' if not ppl_x else ', exceptions ' + ', '.join(ppl_x)}", flush=True)
    print(f"written: {run.out_path}", flush=True)
    print("=== END ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
