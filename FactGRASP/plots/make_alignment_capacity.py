# figure 6 and table 9 from final_results/appendix/alignment_capacity_summary_full.csv

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P

DATA = ROOT / "final_results" / "appendix" / "alignment_capacity_summary_full.csv"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "figures"

PAIR_ORDER = [
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"),
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"),
    ("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"),
    ("Qwen/Qwen3.5-4B", "meta-llama/Llama-3.1-8B"),
    ("meta-llama/Llama-3.1-8B", "answerdotai/ModernBERT-base"),
    ("meta-llama/Llama-3.1-8B", "google/gemma-4-12B"),
]
PAIR_LABEL = {
    PAIR_ORDER[0]: r"Q4B $\leftarrow$ Q0.8B",
    PAIR_ORDER[1]: r"Q4B $\leftarrow$ Q9B",
    PAIR_ORDER[2]: r"Q4B $\leftarrow$ Smol 3B",
    PAIR_ORDER[3]: r"Q4B $\leftarrow$ Llama 8B",
    PAIR_ORDER[4]: r"L8B $\leftarrow$ Modern",
    PAIR_ORDER[5]: r"L8B $\leftarrow$ Gemma 12B",
}
METHOD_ORDER = [
    ("full_ridge", "", "Full ridge", P.CLAY, "o", "-"),
    ("truncated_ridge", "256", "Trunc. ridge 256", P.SLATE, "s", "--"),
    ("truncated_ridge", "64", "Trunc. ridge 64", P.SAGE, "^", ":"),
    ("orthogonal_procrustes", "", "Orthogonal", P.GRAY, "D", "-."),
]
SIZES = [512, 2048, 8192, 33937]
X = list(range(len(SIZES)))


def read_rows() -> list[dict]:
    with DATA.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 96, f"expected 96 summary rows, found {len(rows)}"
    return rows


def index_rows(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        key = (r["A"], r["B"], r["method"], r["rank"], int(r["n_train"]))
        assert key not in by
        by[key] = r
    return by


def denominator(by: dict, pair: tuple[str, str]) -> float:
    return float(by[(pair[0], pair[1], "full_ridge", "", 33937)]["mrr_mean"])


def render_figure(rows: list[dict]) -> None:
    P.apply_style()
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        "font.size": 9.0, "axes.labelsize": 9.0, "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    by = index_rows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.97, bottom=.14, top=.865, wspace=.20, hspace=.24)

    for panel_i, (ax, pair) in enumerate(zip(axes.flat, PAIR_ORDER)):
        denom = denominator(by, pair)
        ax.axhline(1.0, color=P.SUBINK, lw=.65, ls=(0, (2, 2)), zorder=0)
        for method, rank, _, color, marker, linestyle in METHOD_ORDER:
            vals, errs = [], []
            for n in SIZES:
                r = by[(pair[0], pair[1], method, rank, n)]
                vals.append(float(r["mrr_mean"]) / denom)
                sd = r["mrr_sd"]
                errs.append(0.0 if not sd else float(sd) / denom)
            ax.plot(X, vals, color=color, marker=marker, ls=linestyle, lw=1.15,
                    ms=3.7, mfc="white" if method != "full_ridge" else color,
                    mec=color, mew=.75, zorder=3)
            ax.errorbar(X[:3], vals[:3], yerr=errs[:3], fmt="none", ecolor=color,
                        elinewidth=.7, capsize=1.7, capthick=.7, zorder=2)
        ax.text(.0, 1.025, f"({chr(97 + panel_i)})  {PAIR_LABEL[pair]}",
                transform=ax.transAxes, ha="left", va="bottom", color=P.INK,
                clip_on=False)
        ax.set_xlim(-.12, 3.12)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(X, ["512", "2k", "8k", "33.9k"])
        ax.set_yticks([0, .5, 1.0])
        P.softgrid(ax, "y")

    fig.text(.023, .50, "MRR / full-data ridge MRR", ha="center", va="center",
             rotation=90, color=P.INK, fontsize=9)
    fig.text(.55, .035, "paired training queries", ha="center", va="center", color=P.INK,
             fontsize=9)
    handles = [Line2D([], [], color=color, marker=marker, ls=linestyle, lw=1.15,
                      ms=3.7, mfc="white" if method != "full_ridge" else color,
                      mec=color, label=label)
               for method, _, label, color, marker, linestyle in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .985), ncol=4,
               columnspacing=.75, handlelength=1.35, handletextpad=.35, borderaxespad=0)

    overlap_count = P.overlap_check(fig, "figure6_alignment_capacity")
    if overlap_count:
        raise SystemExit(f"figure6_alignment_capacity: {overlap_count} text overlaps")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bbox = matplotlib.transforms.Bbox([[0, 0], [5.5, 3.55]])
    fig.savefig(FIG_DIR / "figure6_alignment_capacity.pdf", bbox_inches=bbox)
    fig.savefig(FIG_DIR / "figure6_alignment_capacity.png", dpi=300, bbox_inches=bbox)
    plt.close(fig)


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%")


def table_value(row: dict, denom: float, include_sd: bool) -> str:
    value = float(row["mrr_mean"]) / denom
    if include_sd:
        sd = float(row["mrr_sd"]) / denom
        return f"${value:.3f} \\pm {sd:.3f}$"
    return f"${value:.3f}$"


def render_table(rows: list[dict]) -> None:
    by = index_rows(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{2.6pt}",
        (r"\caption{Alignment capacity and sample efficiency. Entries at 512, 2,048, "
         r"8,192, and 33,937 are filtered MRR divided by the same pair's full-data "
         r"full-ridge MRR. The first three columns report mean $\pm$ standard deviation "
         r"over three nested-subset permutations; the full-data column is one deterministic "
         r"fit. Truncated ridge retains leading source singular directions and is not "
         r"response-aware reduced-rank regression. The final column gives absolute full-data MRR.}" ),
        r"\label{tab:alignment-capacity}",
        r"\begin{tabular}{@{}ll@{\qquad}ll@{}}",
        r"(a) & Qwen3.5-4B $\leftarrow$ Qwen3.5-0.8B & (d) & Qwen3.5-4B $\leftarrow$ Llama-3.1-8B \\",
        r"(b) & Qwen3.5-4B $\leftarrow$ Qwen3.5-9B & (e) & Llama-3.1-8B $\leftarrow$ ModernBERT-base \\",
        r"(c) & Qwen3.5-4B $\leftarrow$ SmolLM3-3B & (f) & Llama-3.1-8B $\leftarrow$ Gemma-4-12B \\",
        r"\end{tabular}",
        r"\vspace{3pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Target $\leftarrow$ source & Map & 512 & 2,048 & 8,192 & 33,937 & Full MRR \\",
        r"\midrule",
        r"% source: final_results/appendix/alignment_capacity_summary_full.csv",
    ]
    method_tex = {
        ("full_ridge", ""): "Full ridge",
        ("truncated_ridge", "256"): "Trunc. 256",
        ("truncated_ridge", "64"): "Trunc. 64",
        ("orthogonal_procrustes", ""): "Orthogonal",
    }
    pair_plain = {pair: f"({chr(97 + i)})" for i, pair in enumerate(PAIR_ORDER)}
    for pair_i, pair in enumerate(PAIR_ORDER):
        denom = denominator(by, pair)
        for method_i, (method, rank, _, _, _, _) in enumerate(METHOD_ORDER):
            values = [table_value(by[(pair[0], pair[1], method, rank, n)], denom,
                                  include_sd=n != 33937) for n in SIZES]
            absolute = float(by[(pair[0], pair[1], method, rank, 33937)]["mrr_mean"])
            pair_cell = pair_plain[pair] if method_i == 0 else ""
            lines.append(f"{pair_cell} & {method_tex[(method, rank)]} & "
                         + " & ".join(values) + f" & ${absolute:.3f}$ \\\\")
        if pair_i != len(PAIR_ORDER) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "alignment_capacity.tex").write_text("\n".join(lines))


def validate(rows: list[dict]) -> None:
    by = index_rows(rows)
    for pair in PAIR_ORDER:
        denom = denominator(by, pair)
        assert denom > 0
        endpoint = float(by[(pair[0], pair[1], "full_ridge", "", 33937)]["mrr_mean"])
        assert math.isclose(endpoint / denom, 1.0, abs_tol=1e-12)
        for method, rank, *_ in METHOD_ORDER:
            for n in SIZES:
                r = by[(pair[0], pair[1], method, rank, n)]
                assert math.isfinite(float(r["mrr_mean"]))
                if n == 33937:
                    assert r["mrr_sd"] == "" and r["n_subset_draws"] == "1"
                else:
                    assert r["mrr_sd"] != "" and r["n_subset_draws"] == "3"


def main() -> None:
    rows = read_rows()
    validate(rows)
    render_figure(rows)
    render_table(rows)
    print("wrote figures/figure6_alignment_capacity.pdf")
    print("wrote figures/figure6_alignment_capacity.png")
    print("wrote figures/alignment_capacity.tex")


if __name__ == "__main__":
    main()
