# colours fonts and model order shared by the plots

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_check import check as overlap_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "final_results", "data")

TEXT_WIDTH = 5.5
HALF_WIDTH = 2.65

INK = "#20201D"
SUBINK = "#5C574F"
CLAY = "#BF5B3D"
CLAY_DK = "#8F3F28"
CLAY_SOFT = "#E2B7A3"
SLATE = "#456076"
SAGE = "#77854F"
OCHRE = "#C98A2B"
GRAY = "#A9A198"
GRID = "#ECE7DE"
BAND = "#F3E9E1"

COLOR = {"infonce": CLAY, "cosine": SLATE, "mixture": SAGE, "oracle": OCHRE,
         "ceiling": OCHRE, "random": GRAY, "control": GRAY, "threshold": CLAY_DK,
         "single": SLATE, "ink": INK}

LABEL = {"infonce": "InfoNCE objective", "cosine": "Pointwise cosine objective"}

FAMILY_ORDER = ["Qwen3.5", "SmolLM3", "Gemma-4", "Ministral-3", "Llama-3.1", "Granite-4.2",
                "Llama-3.2", "ModernBERT", "OLMo-3", "Olmo-3"]


FAMILY_LABEL = {"Qwen3.5": "Qwen3.5", "SmolLM3": "SmolLM3", "Gemma-4": "Gemma-4",
                "Ministral-3": "Ministral", "Llama-3.1": "Llama-3.1",
                "Granite-4.2": "Granite", "Llama-3.2": "Llama-3.2",
                "ModernBERT": "ModernBERT", "OLMo-3": "Olmo-3", "Olmo-3": "Olmo-3"}


def model_order(models, size_of, family_of):
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(models, key=lambda m: (rank.get(family_of(m), len(rank)),
                                         size_of(m), m))


FAMILY_C = {"Qwen3.5": CLAY, "Gemma-4": OCHRE, "Llama-3.1": SLATE,
            "Llama-3.2": SLATE, "Olmo-3": SAGE, "OLMo-3": SAGE, "SmolLM3": CLAY_DK,
            "Ministral-3": SUBINK, "Granite-4.2": GRAY, "ModernBERT": GRAY,
            "encoder": GRAY}

MODEL_C = {"Qwen3.5-4B": CLAY, "Qwen3.5-9B": CLAY_DK, "SmolLM3-3B": SAGE,
           "Gemma-4-E4B": OCHRE, "Llama-3.1-8B": SLATE, "OLMo-3-7B": SUBINK,
           "encoder": GRAY}
SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
         "google/gemma-4-E4B": "Gemma-4-E4B", "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
         "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
         "google/gemma-4-E2B": "Gemma-4-E2B", "google/gemma-4-12B": "Gemma-4-12B",
         "google/gemma-4-E4B-it": "Gemma-4-E4B-it",
         "google/gemma-4-E2B-it": "Gemma-4-E2B-it", "google/gemma-4-12B-it": "Gemma-4-12B-it",
         "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B-Instruct",
         "allenai/Olmo-3-1025-7B": "Olmo-3-7B",
         "answerdotai/ModernBERT-base": "ModernBERT-base",
         "ibm-granite/granite-4.2-8b": "Granite-4.2-8B",
         "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B", "Qwen/Qwen3.5-2B": "Qwen3.5-2B",
         "Qwen/Qwen3.5-4B-Base": "Qwen3.5-4B-Base",
         "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
         "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
         "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
         "allenai/Olmo-3-1025-7B": "Olmo-3-7B",
         "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B-Instruct",
         "bert-base-uncased": "BERT-base",
         "mistralai/Ministral-3-8B-Instruct-2512-BF16": "Ministral-3-8B"}


AXIS_LABEL = {"Qwen/Qwen3.5-0.8B": "Qwen 0.8B", "Qwen/Qwen3.5-2B": "Qwen 2B",
              "Qwen/Qwen3.5-4B-Base": "Qwen 4B-base", "Qwen/Qwen3.5-4B": "Qwen 4B",
              "Qwen/Qwen3.5-9B": "Qwen 9B", "HuggingFaceTB/SmolLM3-3B": "SmolLM3 3B",
              "google/gemma-4-E2B": "Gemma E2B", "google/gemma-4-E4B": "Gemma E4B",
              "google/gemma-4-E4B-it": "Gemma E4B-it", "google/gemma-4-12B": "Gemma 12B",
              "mistralai/Ministral-3-8B-Instruct-2512-BF16": "Ministral 8B",
              "meta-llama/Llama-3.1-8B": "Llama 8B",
              "meta-llama/Llama-3.1-8B-Instruct": "Llama 8B-it",
              "meta-llama/Llama-3.2-1B": "Llama 1B", "meta-llama/Llama-3.2-3B": "Llama 3B",
              "ibm-granite/granite-4.2-8b": "Granite 8B",
              "answerdotai/ModernBERT-base": "ModernBERT",
              "allenai/Olmo-3-1025-7B": "Olmo 7B", "allenai/Olmo-3-7B-Instruct": "Olmo 7B-it",
              "bert-base-uncased": "BERT-base"}


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "Times",
                       "serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif", "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold", "mathtext.sf": "serif",
        "mathtext.cal": "serif:italic", "mathtext.default": "it",
        "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.7, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.2, "legend.handletextpad": 0.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "lines.solid_capstyle": "round", "errorbar.capsize": 2,
        "patch.linewidth": 0.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def softgrid(axis_obj, axis="y"):
    axis_obj.grid(True, axis=axis, color=GRID, linewidth=0.8, zorder=0)
    axis_obj.set_axisbelow(True)


def panel(axis_obj, letter):
    axis_obj.set_title("(%s)" % letter, color=INK, loc="left", pad=4)


def read(name):
    with open(os.path.join(DATA, name)) as handle:
        return list(csv.DictReader(handle))


def exists(name):
    return os.path.exists(os.path.join(DATA, name))


def save(fig, name, tight=True):
    count = overlap_check(fig, name)
    if count:
        raise SystemExit("%s: %d text overlaps" % (name, count))
    path = os.path.join(OUT, name + ".pdf")
    if tight:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    else:
        width, height = fig.get_size_inches()
        fig.savefig(path, bbox_inches=Bbox([[0.0, 0.0], [width, height]]))
    plt.close(fig)
    print("wrote %s" % os.path.relpath(path, ROOT))
