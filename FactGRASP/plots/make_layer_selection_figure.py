# figure 5 and table 7 from final_results/appendix

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

FONT = "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(BOLD)
mpl.rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_results" / "appendix"
VALIDATION = FINAL / "independent_layer_selection_validation_curves.csv"
TEST = FINAL / "independent_layer_selection_test_metrics.csv"
SUMMARY = FINAL / "independent_layer_selection_summary.json"
PAIRED = FINAL / "independent_layer_selection_paired_differences.json"
STATE = FINAL / "independent_layer_selection_state.json"
DESIGN = FINAL / "independent_layer_selection_design.json"
PAPER_FIG = ROOT / "figures" / "figure5_layer_selection.pdf"
PAPER_FIG_PNG = ROOT / "figures" / "figure5_layer_selection.png"
PAPER_TABLE = ROOT / "figures" / "independent_layer_selection.tex"

PAIR_ORDER = ["bert_wikidata", "bert_conceptnet", "qwen4b_wikidata",
              "gemmae4b_wikidata"]
TITLES = {
    "bert_wikidata": "BERT · Wikidata",
    "bert_conceptnet": "BERT · ConceptNet",
    "qwen4b_wikidata": "Qwen3.5-4B · Wikidata",
    "gemmae4b_wikidata": "Gemma-4-E4B · Wikidata",
}
RECIPES = ["single_cosine", "mixture8_cosine", "single_composite_infonce",
           "mixture8_composite_infonce"]
LABELS = {
    "single_cosine": "single, cosine",
    "mixture8_cosine": "mixture, cosine",
    "single_composite_infonce": "single, composite",
    "mixture8_composite_infonce": "mixture, composite",
}
STYLES = {
    "single_cosine": ("#456076", "-", "o"),
    "mixture8_cosine": ("#456076", "--", "s"),
    "single_composite_infonce": ("#BF5B3D", "-", "o"),
    "mixture8_composite_infonce": ("#BF5B3D", "--", "s"),
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def figure() -> None:
    rows = read_csv(VALIDATION)
    chosen = json.load(open(SUMMARY))["selected_layers"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.35), constrained_layout=False)
    for ax, pair in zip(axes.flat, PAIR_ORDER):
        pair_rows = [r for r in rows if r["pair"] == pair]
        for recipe in RECIPES:
            rr = sorted((r for r in pair_rows if r["recipe"] == recipe),
                        key=lambda r: int(r["layer"]))
            x = [int(r["layer"]) for r in rr]
            y = [float(r["MRR"]) for r in rr]
            color, ls, marker = STYLES[recipe]
            ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                    markersize=3.5, linewidth=1.35, label=LABELS[recipe])
            selected = int(chosen[pair][recipe])
            j = x.index(selected)
            ax.scatter([x[j]], [y[j]], s=38, facecolors="none", edgecolors=color,
                       linewidths=1.2, zorder=5)
        ax.set_title(TITLES[pair], fontweight="normal", pad=3)
        ax.set_xticks(sorted({int(r["layer"]) for r in pair_rows}))
        ax.grid(axis="y", color="#ECE7DE", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("hidden-state layer")
        ax.set_ylabel("validation MRR")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.002), columnspacing=1.3, handlelength=2.3)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.19,
                        hspace=0.48, wspace=0.30)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FINAL / f"independent_layer_validation_curves.{ext}", **kw)
    plt.close(fig)
    caption = (
        "Validation layer selection on the predeclared five-point coarse grid. Each line "
        "uses selection seed 1729 and the same 8,000-row calibration set; open circles "
        "mark the validation-selected layer. Single and eight-expert mixture probes use "
        "cosine or composite cosine+InfoNCE training. Final test estimates use fresh "
        "training seeds 0, 1, and 2; no test ranks or test candidate pool enter selection."
    )
    (FINAL / "independent_layer_validation_curves_caption.txt").write_text(caption + "\n")
    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL / "independent_layer_validation_curves.pdf", PAPER_FIG)
    shutil.copyfile(FINAL / "independent_layer_validation_curves.png", PAPER_FIG_PNG)


def appendix_table() -> None:
    test = read_csv(TEST)
    chosen = json.load(open(SUMMARY))["selected_layers"]
    paired = json.load(open(PAIRED))
    lookup = {(r["pair"], r["role"], r["recipe"]): r for r in test}
    out = []
    for pair in PAIR_ORDER:
        for recipe in RECIPES:
            fixed = lookup[(pair, "fixed_shared", recipe)]
            indep = lookup[(pair, "independent", recipe)]
            gap = paired[pair]["layer_selection_gain"][recipe]
            out.append({
                "pair": pair,
                "recipe": LABELS[recipe],
                "fixed_layer": int(fixed["layer"]),
                "selected_layer": int(chosen[pair][recipe]),
                "fixed_MRR_mean": float(fixed["MRR_mean"]),
                "fixed_MRR_seed_sd": float(fixed["MRR_sd"]),
                "independent_MRR_mean": float(indep["MRR_mean"]),
                "independent_MRR_seed_sd": float(indep["MRR_sd"]),
                "independent_minus_fixed": float(gap["diff"]),
                "paired_ci_low": float(gap["ci_low"]),
                "paired_ci_high": float(gap["ci_high"]),
            })
    csv_path = FINAL / "independent_layer_selection_appendix_table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    pair_names = {
        "bert_wikidata": "BERT / Wikidata",
        "bert_conceptnet": "BERT / ConceptNet",
        "qwen4b_wikidata": "Qwen3.5-4B / Wikidata",
        "gemmae4b_wikidata": "Gemma-4-E4B / Wikidata",
    }
    recipe_names = {
        "single, cosine": "single / cos.",
        "mixture, cosine": "mix. / cos.",
        "single, composite": "single / comp.",
        "mixture, composite": "mix. / comp.",
    }
    lines = [
        "% Source: final_results/appendix/independent_layer_selection_test_metrics.csv",
        "%         final_results/appendix/independent_layer_selection_paired_differences.json",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Independent validation-layer selection on a predeclared five-depth grid. "
        "$\\ell_0$ is the fixed shared layer and $\\hat\\ell$ is the validation-selected "
        "layer. MRR entries are mean $\\pm$ seed SD for full-training seeds 0, 1, and 2. "
        "$\\Delta$ is selected minus the same-code fixed-layer reference; intervals are "
        "paired bootstrap 95\\% intervals over test triples after averaging reciprocal "
        "rank across seeds. Selection used seed 1729 only.}",
        "\\label{tab:independent-layer-selection}",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{llrrccc}",
        "\\toprule",
        "Setting & Recipe & $\\ell_0$ & $\\hat\\ell$ & Fixed MRR & Selected MRR & $\\Delta$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    last = None
    for r in out:
        if last is not None and r["pair"] != last:
            lines.append("\\addlinespace[2pt]")
        setting = pair_names[r["pair"]] if r["pair"] != last else ""
        fixed = f"{r['fixed_MRR_mean']:.3f} $\\pm$ {r['fixed_MRR_seed_sd']:.3f}"
        indep = f"{r['independent_MRR_mean']:.3f} $\\pm$ {r['independent_MRR_seed_sd']:.3f}"
        gap = (f"{r['independent_minus_fixed']:+.3f} "
               f"[{r['paired_ci_low']:+.3f}, {r['paired_ci_high']:+.3f}]")
        lines.append(f"{setting} & {recipe_names[r['recipe']]} & {r['fixed_layer']} & "
                     f"{r['selected_layer']} & {fixed} & {indep} & {gap} \\\\")
        last = r["pair"]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    table_text = "\n".join(lines)
    (FINAL / "independent_layer_selection_appendix_table.tex").write_text(table_text)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text(table_text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def metadata() -> None:
    state = json.load(open(STATE))
    selection_cells = [v for pair in state["selection"].values() for v in pair.values()]
    final_cells = [v for pair in state["final"].values() for v in pair.values()]
    sources = {}
    for source in state.get("feature_sources", {}).values():
        sources[source] = sources.get(source, 0) + 1
    meta = {
        "experiment": "independent_layer_selection",
        "design_sha256": sha256(DESIGN),
        "script_sha256": sha256(Path(__file__).with_name("independent_layer_selection.py")),
        "pairs": PAIR_ORDER,
        "selection_cells": len(selection_cells),
        "unique_final_cells": len(final_cells),
        "selection_training_seconds": sum(float(x["seconds"]) for x in selection_cells),
        "final_training_seconds": sum(float(x["seconds"]) for x in final_cells),
        "feature_sources": sources,
        "gpu": "physical 0, NVIDIA RTX A6000",
        "validation": {
            "static_compile": "pass",
            "smoke_cells": 8,
            "smoke_rr_metric_consistency": "pass",
            "resumability_rerun": "pass",
            "selection_complete": len(selection_cells) == 80,
            "final_complete": len(final_cells) == 63,
            "ordered_triple_hashes_recorded": True,
            "metric_gold_pool_alignment_assertions": "pass",
            "test_ranks_or_test_candidate_pool_used_for_selection": False,
            "test_triples_used_only_as_standard_known-positive filter": True,
        },
        "artifacts": {
            "figure_pdf": str(PAPER_FIG),
            "appendix_table": str(PAPER_TABLE),
            "raw_state": str(STATE),
            "test_metrics": str(TEST),
            "raw_seed_metrics": str(FINAL / "independent_layer_selection_raw_seeds.csv"),
            "paired_differences": str(PAIRED),
        },
    }
    with open(FINAL / "independent_layer_selection_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def raw_seed_table() -> None:
    state = json.load(open(STATE))
    rows = []
    for pair, cells in state["final"].items():
        for key, meta in cells.items():
            d = __import__("torch").load(meta["cell"], map_location="cpu",
                                         weights_only=False)
            rows.append({
                "pair": pair,
                "recipe": d["recipe"],
                "layer": d["layer"],
                "seed": d["seed"],
                "roles": ";".join(meta["roles"]),
                "MRR": d["metrics"]["MRR"],
                "Hits@1": d["metrics"]["Hits@1"],
                "Hits@3": d["metrics"]["Hits@3"],
                "Hits@10": d["metrics"]["Hits@10"],
                "epochs_run": d["epochs_run"],
                "best_epoch": d["best_epoch"],
                "seconds": d["seconds"],
                "cell": meta["cell"],
            })
    rows.sort(key=lambda r: (PAIR_ORDER.index(r["pair"]), RECIPES.index(r["recipe"]),
                             r["layer"], r["seed"]))
    with open(FINAL / "independent_layer_selection_raw_seeds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    figure()
    appendix_table()
    # these two read the raw per seed checkpoints, they only exist after running the experiment yourself
    if (ROOT / "results" / "appendix" / "independent_layer_selection").exists():
        raw_seed_table()
        metadata()
    print("wrote independent-layer figure and appendix table")


if __name__ == "__main__":
    main()
