import os

# every path is relative to the repo root so the code runs from a fresh clone
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = ROOT
KG_ROOT = ROOT
KG_CODE = HERE
KG_DATA = os.path.join(ROOT, "data")
KG_CACHE = os.path.join(ROOT, "cache")
CACHE_DEC = os.path.join(KG_CACHE, "decoders")
LOES_ROOT = os.path.join(ROOT, "loes")
RUNS = os.path.join(ROOT, "results", "runs")
DATA_EXTRA = os.path.join(ROOT, "data_extra")
LOGS = os.path.join(ROOT, "logs")

# hidden state caches and raw run json are big and not in the repo, they get made on first run
for _d in (KG_CACHE, CACHE_DEC, RUNS, LOGS):
    os.makedirs(_d, exist_ok=True)
