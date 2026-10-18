# figures 3 4 7 and 8 from the csv files in final_results/data, writes to figures/
# this is the script version used for the submitted pdf so labels like filtered mrr are as submitted
# run: python plots/make_paper_figures.py

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P

DATA = ROOT / "final_results" / "data"
REV_DATA = ROOT / "final_results" / "appendix"
OUT = ROOT / "figures"
AUDIT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

INK, SUBINK, CLAY, CLAY_DK = P.INK, P.SUBINK, P.CLAY, P.CLAY_DK
SLATE, SAGE, OCHRE, GRAY = P.SLATE, P.SAGE, P.OCHRE, P.GRAY
GRID, PAPER = P.GRID, "#FBF8F3"
WARM = colors.LinearSegmentedColormap.from_list("warm", [PAPER, "#E7D3C7", CLAY, CLAY_DK])
DIVERGE = colors.LinearSegmentedColormap.from_list("residual", [SLATE, PAPER, CLAY])


def read(name: str, revision: bool = False) -> list[dict]:
    path = (REV_DATA if revision else DATA) / name
    with path.open() as fh:
        return list(csv.DictReader(fh))


def short(model: str) -> str:
    known = dict(P.SHORT)
    known.update({
        "tiiuae/falcon-mamba-7b": "Falcon-Mamba 7B",
        "answerdotai/ModernBERT-base": "ModernBERT",
        "allenai/Olmo-Hybrid-7B": "Olmo-Hybrid base",
        "allenai/Olmo-Hybrid-Instruct-SFT-7B": "Olmo-Hybrid SFT",
        "allenai/Olmo-Hybrid-Instruct-DPO-7B": "Olmo-Hybrid DPO",
    })
    return known.get(model, model.split("/")[-1].replace("-Instruct", "-it"))


def axis_short(model: str) -> str:
    return P.AXIS_LABEL.get(model, short(model)).replace("-", " ")


def family_label(name: str) -> str:
    return {"OLMo-3": "Olmo-3", "FalconMamba": "Falcon-Mamba",
            "Granite-4.2": "Granite", "Ministral-3": "Ministral"}.get(name, name)


def style() -> None:
    P.apply_style()
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"],
        "mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold", "mathtext.sf": "Liberation Serif",
        "font.size": 9.0, "axes.labelsize": 9.0, "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def clean(ax, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)


def panel(ax, letter: str, x: float = 0.0, y: float = 1.02) -> None:
    draw = ax.text2D if hasattr(ax, "text2D") else ax.text
    draw(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", color=INK)


OVERLAPS: dict[str, int] = {}


def save(fig, name: str) -> None:
    count = P.overlap_check(fig, name)
    OVERLAPS[name] = count
    if count:
        print(f"warning: {name} overlap checker reports {count}")
    w, h = fig.get_size_inches()
    bbox = matplotlib.transforms.Bbox([[0, 0], [w, h]])
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=bbox)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches=bbox)
    plt.close(fig)


def family_order_s7(rows: list[dict]) -> tuple[list[str], dict[str, str]]:
    fam = {}
    for r in rows:
        fam[r["A"]] = r["family_A"]
        fam[r["B"]] = r["family_B"]
    rank = {f: i for i, f in enumerate(P.FAMILY_ORDER)}
    models = sorted(fam, key=lambda m: (rank.get(fam[m], 99), short(m)))
    return models, fam


def block_outlines(ax, models: list[str], fam: dict[str, str]) -> list[tuple[int, int, str]]:
    blocks = []
    start = 0
    for i in range(1, len(models) + 1):
        if i == len(models) or fam[models[i]] != fam[models[start]]:
            blocks.append((start, i, fam[models[start]]))
            ax.add_patch(Rectangle((start - .5, start - .5), i - start, i - start,
                                   fill=False, ec=INK, lw=.5))
            start = i
    return blocks


def figure3_transfer() -> list[str]:
    pairs = read("transfer_all_pairs.csv")
    models, fam = family_order_s7(pairs)
    idx = {m: i for i, m in enumerate(models)}
    mat = np.full((len(models), len(models)), np.nan)
    for r in pairs:
        mat[idx[r["A"]], idx[r["B"]]] = float(r["tau"])

    np.fill_diagonal(mat, np.nan)
    masked = np.ma.masked_invalid(mat)
    cmap = DIVERGE.copy(); cmap.set_bad("#C8C3BC")
    fig = plt.figure(figsize=(5.5, 3.35))
    ax = fig.add_axes([.055, .27, .50, .66])
    im = ax.imshow(masked, vmin=.7, vmax=1.3, cmap=cmap, aspect="equal", interpolation="nearest")
    block_outlines(ax, models, fam)
    ticks = np.arange(len(models))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=90)
    ax.set_yticks(ticks, [str(i + 1) for i in ticks])
    ax.set_xlabel("input model index", labelpad=2)
    ax.set_ylabel("reference probe index", labelpad=2)
    panel(ax, "a", -.13, 1.02)
    ax.text(.5, 1.025, f"median {np.nanmedian(mat):.3f}", transform=ax.transAxes,
            ha="center", va="bottom")
    cax = fig.add_axes([.12, .045, .37, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([.7, 1.0, 1.3])
    cb.set_label("MRR ratio to reference, centered at 1", labelpad=1)
    cb.ax.xaxis.set_label_position("top")

    pool = read("reverse_map.csv")
    pool.sort(key=lambda r: float(r["hub_matched_own_pool"]))
    ax2 = fig.add_axes([.72, .12, .265, .75])
    y = np.arange(len(pool))
    ref = np.array([float(r["hub"]) for r in pool])
    returned = np.array([float(r["hub_matched_own_pool"]) for r in pool])
    for yi, a, b in zip(y, ref, returned):
        ax2.plot([b, a], [yi, yi], color=GRID, lw=.8, zorder=1)
    ax2.scatter(ref, y, s=16, facecolors="white", edgecolors=CLAY, lw=.8, zorder=3,
                label="reference candidate space")
    ax2.scatter(returned, y, s=16, color=SLATE, zorder=3, label="returned to input space")
    labels = [axis_short(r["model"]) + ("*" if r["model"] == "answerdotai/ModernBERT-base" else "")
              for r in pool]
    ax2.set_yticks(y, labels)
    ax2.tick_params(axis="y", length=0, pad=2, labelsize=9)
    ax2.set_xlabel("filtered MRR", labelpad=2)
    ax2.set_ylabel("")
    ax2.set_xlim(0, .54)
    clean(ax2, "x")
    panel(ax2, "b", -.16, 1.02)
    ax2.legend(loc="lower right", bbox_to_anchor=(1, 1.015), fontsize=9,
               handlelength=1.0, borderaxespad=0, labelspacing=.2)
    save(fig, "figure3_transfer")
    return [f"{i+1}: {short(m)} [{family_label(fam[m])}]" for i, m in enumerate(models)]


def figure4_training_agreement() -> list[str]:
    rows = [r for r in read("olmo2_checkpoints.csv") if r["hub"] == "meta-llama/Llama-3.1-8B"]
    staged = [r for r in rows if r["revision"] != "main"]
    staged.sort(key=lambda r: float(r["tokens_B"]))
    released = next(r for r in rows if r["revision"] == "main")
    x = np.array([float(r["tokens_B"]) for r in staged])
    native = np.array([float(r["native_MRR"]) for r in staged])
    transfer = np.array([float(r["hub_MRR_zero_label"]) for r in staged])

    fig = plt.figure(figsize=(5.5, 2.6))
    ax = fig.add_axes([.10, .19, .42, .73])
    floor_lo = min(float(rows[0]["floor_MRR"]), float(rows[0]["hub_floor_MRR"]))
    floor_hi = max(float(rows[0]["floor_MRR"]), float(rows[0]["hub_floor_MRR"]))
    ax.axhspan(floor_lo, floor_hi, color=GRID, zorder=0)
    ax.plot(x[:2], native[:2], color=SLATE, marker="o", mfc="white", mec=SLATE, lw=1.1)
    ax.plot(x[2:], native[2:], color=SLATE, marker="o", mfc="white", mec=SLATE, lw=1.1)
    ax.plot(x[:2], transfer[:2], color=CLAY, marker="o", lw=1.1)
    ax.plot(x[2:], transfer[2:], color=CLAY, marker="o", lw=1.1)
    ax.scatter([4800], [float(released["native_MRR"])], marker="D", s=23,
               facecolors="white", edgecolors=SLATE, zorder=4)
    ax.scatter([4800], [float(released["hub_MRR_zero_label"])], marker="D", s=23,
               color=CLAY, zorder=4)
    ax.text(21, .303, "first clear rise: 21B", ha="left", va="top", color=INK)
    ax.annotate("released checkpoint", xy=(4800, float(released["native_MRR"])),
                xytext=(-55, 22), textcoords="offset points", ha="right", va="bottom",
                color=INK, arrowprops={"arrowstyle": "-", "color": SUBINK, "lw": .6})
    ax.set_xscale("symlog", linthresh=1, linscale=.7, base=10)
    ax.set_xlim(-.15, 7000)
    ax.set_ylim(.175, .415)
    ax.set_xticks([0, 1, 21, 100, 1000, 4800], ["0", "1", "21", "100", "1k", "released"])
    ax.set_yticks([.2, .3, .4])
    ax.set_xlabel("pretraining tokens (billions; symmetric log scale)", labelpad=2)
    ax.set_ylabel("filtered MRR", labelpad=2)
    clean(ax, "y")
    panel(ax, "a", -.16, 1.01)
    ax.legend(handles=[Line2D([], [], color=SLATE, marker="o", mfc="white", label="native"),
                       Line2D([], [], color=CLAY, marker="o", label="transferred probe")],
              loc="upper left", bbox_to_anchor=(.02, .98), ncol=1,
              handlelength=1.2, borderaxespad=0, labelspacing=.25)

    ag = read("fact_agreement_matrix.csv")
    models = [r["model"] for r in ag]
    fam = {r["model"]: r["family"] for r in ag}
    rank = {f: i for i, f in enumerate(P.FAMILY_ORDER + ["FalconMamba"])}
    models.sort(key=lambda m: (rank.get(fam[m], 99), short(m)))
    by = {r["model"]: r for r in ag}
    mat = np.array([[float(by[a]["spearman__" + b]) for b in models] for a in models])
    ax2 = fig.add_axes([.63, .30, .34, .60])
    im = ax2.imshow(mat, vmin=0, vmax=1, cmap=WARM, interpolation="nearest")
    block_outlines(ax2, models, fam)
    ticks = [0, 5, 10, 15, 19]
    ax2.set_xticks(ticks, [str(i + 1) for i in ticks])
    ax2.set_yticks(ticks, [str(i + 1) for i in ticks])
    ax2.set_xlabel("model index", labelpad=2)
    ax2.set_ylabel("model index", labelpad=2)
    panel(ax2, "b", -.2, 1.01)
    cax = fig.add_axes([.66, .080, .28, .022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, .5, 1])
    cb.set_label("fact-level rank agreement", labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    save(fig, "figure4_training_agreement")
    return [f"{i+1}: {short(m)} [{family_label(fam[m])}]" for i, m in enumerate(models)]


def figure7_post_training() -> None:
    pairs = read("post_training_pairs.csv")
    pairs.sort(key=lambda r: float(r["readability_delta"]))
    fig = plt.figure(figsize=(5.5, 2.7))
    ax = fig.add_axes([.17, .17, .39, .75])
    y = np.arange(len(pairs))
    for yi, r in zip(y, pairs):
        a, b = float(r["readability_base"]), float(r["readability_post"])
        ax.plot([a, b], [yi, yi], color=GRID, lw=2)
        ax.scatter(a, yi, facecolors="white", edgecolors=SLATE, s=24, zorder=3)
        ax.scatter(b, yi, color=CLAY, s=24, zorder=3)
    ax.set_yticks(y, [r["pair"].replace("-7B", "") for r in pairs])
    ax.set_xlim(.18, .53); ax.set_xticks([.2, .3, .4, .5])
    ax.set_xlabel("filtered MRR", labelpad=2)
    ax.tick_params(axis="y", length=0)
    clean(ax, "x"); panel(ax, "a", -.42, 1.01)
    ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=SLATE, color="none", label="base"),
                       Line2D([], [], marker="o", color=CLAY, ls="none", label="post-trained")],
              loc="lower center", bbox_to_anchor=(.55, 1.02), ncol=2,
              handletextpad=.2, borderaxespad=0)

    traj = read("olmo3_post_training_trajectory.csv")
    ax2 = fig.add_axes([.67, .17, .30, .75])
    for family, color, marker in [("Olmo-3", SLATE, "o"), ("Olmo-Hybrid", CLAY, "D")]:
        rr = sorted([r for r in traj if r["family"] == family], key=lambda r: int(r["stage_order"]))
        xx = np.array([int(r["stage_order"]) for r in rr])
        yy = np.array([float(r["MRR_mixture_infonce"]) for r in rr])
        ax2.plot(xx, yy, color=color, marker=marker, lw=1.2, label=family)
    ax2.set_xticks([0, 1, 2, 3], ["base", "SFT", "DPO", "RLVR"], rotation=25, ha="right")
    ax2.set_ylim(.37, .455); ax2.set_yticks([.38, .42, .45])
    ax2.set_ylabel("filtered MRR", labelpad=2)
    clean(ax2, "y"); panel(ax2, "b", -.24, 1.01)
    ax2.legend(loc="upper right", handlelength=1.0)
    save(fig, "figure7_post_training")


def figure8_relation_expertise() -> list[str]:
    rows = [r for r in read("relation_expertise.csv") if r["graph"] == "wikidata"]
    models = sorted({r["model"] for r in rows}, key=short)
    relations = sorted({r["relation"] for r in rows})
    val = {(r["model"], r["relation"]): float(r["MRR"]) for r in rows}
    weight = {r["relation"]: float(r["n"]) for r in rows if r["model"] == models[0]}
    relations.sort(key=lambda q: np.mean([val[(m, q)] for m in models]), reverse=True)
    mat = np.array([[val[(m, q)] for q in relations] for m in models])
    w = np.array([weight[q] for q in relations], dtype=float); w /= w.sum()
    row_mean = mat @ w
    col_mean = mat.mean(axis=0)
    grand = float(row_mean.mean())
    resid = mat - row_mean[:, None] - col_mean[None, :] + grand
    total = float(np.sum(w[None, :] * (mat - grand) ** 2))
    interaction = float(np.sum(w[None, :] * resid ** 2) / total)
    assert abs(interaction - .041) < .003

    fig = plt.figure(figsize=(5.5, 3.7))
    ax = fig.add_axes([.20, .31, .48, .63])
    im = ax.imshow(mat, aspect="auto", cmap=WARM, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(models)), [axis_short(m) for m in models])
    ax.set_xticks([0, 15, 30, 45, 61], ["1", "16", "31", "46", "62"])
    ax.set_xlabel("relations, easiest to hardest", labelpad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    panel(ax, "a", -.38, 1.01)
    ax2 = fig.add_axes([.72, .31, .22, .63])
    vmax = max(abs(resid.min()), abs(resid.max()))
    im2 = ax2.imshow(resid, aspect="auto", cmap=DIVERGE, vmin=-vmax, vmax=vmax,
                     interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xticks([0, 30, 61], ["1", "31", "62"])
    ax2.set_xlabel("same order", labelpad=2)
    panel(ax2, "b", -.12, 1.01)
    fig.text(.94, .955, f"interaction: {100*interaction:.1f}%", ha="right", va="bottom", color=INK)
    cax = fig.add_axes([.22, .065, .43, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, .5, 1]); cb.set_label("relation-level filtered MRR", labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    cax2 = fig.add_axes([.73, .065, .20, .018])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_ticks([-vmax, 0, vmax]); cb2.set_ticklabels([f"{-vmax:.2f}", "0", f"{vmax:.2f}"])
    cb2.set_label("additive residual MRR", labelpad=1)
    cb2.ax.xaxis.set_label_position("top")
    save(fig, "figure8_relation_expertise")
    return [f"{i+1}: {q}" for i, q in enumerate(relations)]


def write_audit(transfer_map: list[str], agreement_map: list[str], relation_map: list[str]) -> None:
    rows = [
        ("figure3_transfer", "transfer_all_pairs.csv; reverse_map.csv", "5.5 x 3.35",
         "342 directed off-diagonal transfers; one fitted alignment per pair. Pool comparison uses the Llama reference probe."),
        ("figure4_training_agreement", "olmo2_checkpoints.csv; fact_agreement_matrix.csv", "5.5 x 2.6",
         "Three-seed checkpoint means; the released checkpoint is separate; agreement uses 20 models."),
        ("figure7_post_training", "post_training_pairs.csv; olmo3_post_training_trajectory.csv", "5.5 x 2.7",
         "Seven matched base/post pairs; trajectory rows are model-stage evaluations."),
        ("figure8_relation_expertise", "relation_expertise.csv", "5.5 x 3.7",
         "19 models by 62 Wikidata relations; relation-weighted additive interaction share recomputed as 4.1%."),
    ]
    with (AUDIT / "figure_source_audit.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["figure", "released_sources", "final_size_inches", "scope_and_unit"])
        w.writerows(rows)
    text = ["# Figure sources and model index", "",
            "All four PDFs generated by make_paper_figures.py are authored at 5.5 inches wide, use embedded Times-compatible Type 42 fonts, and have matching 300 dpi PNGs. No panel interpolates between models. The overlap counts below come from the required programmatic checker and are followed by raster inspection.", "",
            "## Model index for transfer composite", "", *[f"- {x}" for x in transfer_map], "",
            "## Model index for training agreement", "", *[f"- {x}" for x in agreement_map], "",
            "## Relation order for expertise matrix", "", *[f"- {x}" for x in relation_map], "",
            "## Programmatic overlap check", "", *[f"- {k}: {v}" for k, v in OVERLAPS.items()], ""]
    (AUDIT / "figure_model_index.md").write_text("\n".join(text) + "\n")


def main() -> int:
    style()
    transfer_map = figure3_transfer()
    agreement_map = figure4_training_agreement()
    figure7_post_training()
    relation_map = figure8_relation_expertise()
    write_audit(transfer_map, agreement_map, relation_map)
    print(f"wrote 4 PDF and 4 PNG figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
