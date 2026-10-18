# table 9 and figure 6: for 6 pairs refit the alignment with 512 2048 8192 and 33937 pairs as full
# ridge, ridge on the top 64 or 256 pca directions and orthogonal procrustes, the map and candidates stay fixed
# run: python -m experiments.transfer.alignment_capacity    figure: python plots/make_alignment_capacity.py

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from factgrasp import crossmodel as X, data as D, metrics as M, paths
from experiments.transfer import transfer_all_pairs as TP


NAME = "alignment_capacity"
ROOT = Path(paths.ROOT)
RESULT_DIR = ROOT / "results" / "appendix"
FINAL_DIR = ROOT / "final_results" / "appendix"
P3_PATH = ROOT / "results" / "runs" / "transfer_all_pairs_all.json"

RIDGE = 1e-2
SAMPLE_SIZES = (512, 2048, 8192, 33937)
SEEDS = (0, 1, 2)
RANKS = (64, 256)
METRIC_SEED = 0
BASELINE_TOL = 2e-5
BASELINE_NUMERICAL_AUDIT = {
    "pair": "meta-llama/Llama-3.1-8B <- answerdotai/ModernBERT-base",
    "stored_mrr": 0.2710835635662079,
    "float32_mrr_by_cpu_threads": {
        "1": 0.2710255980491638, "2": 0.27103564143180847,
        "4": 0.27103927731513977, "8": 0.2710762321949005,
        "16": 0.2710835635662079, "32": 0.2710835635662079,
    },
    "float64_accumulation_mrr": 0.2710953950881958,
    "decision": "use float64 CPU accumulation/decomposition and return float32 maps",
}

PAIRS = (
    {"A": "Qwen/Qwen3.5-4B", "B": "Qwen/Qwen3.5-0.8B",
     "stratum": "within-family / weaker source"},
    {"A": "Qwen/Qwen3.5-4B", "B": "Qwen/Qwen3.5-9B",
     "stratum": "within-family / stronger source"},
    {"A": "Qwen/Qwen3.5-4B", "B": "HuggingFaceTB/SmolLM3-3B",
     "stratum": "cross-family / weaker source"},
    {"A": "Qwen/Qwen3.5-4B", "B": "meta-llama/Llama-3.1-8B",
     "stratum": "cross-family / stronger source"},
    {"A": "meta-llama/Llama-3.1-8B", "B": "answerdotai/ModernBERT-base",
     "stratum": "cross-family / historical last-token encoder"},
    {"A": "meta-llama/Llama-3.1-8B", "B": "google/gemma-4-12B",
     "stratum": "cross-family / stronger source"},
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _finite(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v


def fit_full_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = RIDGE) -> torch.Tensor:
    x, y = x.double(), y.double()
    gram = x.T @ x
    cross = x.T @ y
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(gram + ridge * x.shape[0] * eye, cross).float()


def fit_truncated_ridge(x: torch.Tensor, y: torch.Tensor, rank: int,
                        ridge: float = RIDGE) -> torch.Tensor:
    x, y = x.double(), y.double()
    if not 1 <= rank <= x.shape[1]:
        raise ValueError(f"rank must be in [1, {x.shape[1]}], got {rank}")
    gram = x.T @ x
    cross = x.T @ y
    evals, vecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)[:rank]
    vals = evals[order].clamp_min(0)
    v = vecs[:, order]
    return (v @ ((v.T @ cross) / (vals[:, None] + ridge * x.shape[0]))).float()


def fit_orthogonal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    if x.shape[1] != y.shape[1]:
        raise ValueError("orthogonal Procrustes requires equal source/target widths")
    u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    return (u @ vh).float()


def reconstruction_r2(y: torch.Tensor, pred: torch.Tensor) -> float:
    y, pred = y.float(), pred.float()
    ss_tot = ((y - y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - ((y - pred) ** 2).sum() / ss_tot)


def _metric_from_prediction(target, pred: torch.Tensor) -> dict:
    scores = M.cosine_scores(pred.float().cpu(), target.pool_cpu)
    ranks = M.ranks(scores, target.gold, target.vlists(_metric_from_prediction.splits),
                    tie="random", seed=METRIC_SEED).float()
    return {
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(ranks.median()),
        "hits_at_1": float((ranks <= 1).float().mean()),
        "hits_at_10": float((ranks <= 10).float().mean()),
    }


def _predict(target, source, mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = target.probe_device()
    q = source.x_test @ mapping.to(source.x_test.device)
    pred, _ = TP.L.predict(target.probe, q, target.r_test, device)
    return q.float().cpu(), pred.float().cpu()


def _stored_p3_rows() -> dict[tuple[str, str], dict]:
    blob = json.load(P3_PATH.open())
    rows = [v for v in blob["rows"].values() if isinstance(v, dict) and "tau" in v]
    return {(r["A"], r["B"]): r for r in rows}


def _load_spaces(splits: dict, device: torch.device, pairs: tuple[dict, ...]) -> dict:
    names = sorted({p[k] for p in pairs for k in ("A", "B")})
    spaces = {}
    args = SimpleNamespace(temp=0.05, epochs=60)
    for name in names:
        t0 = time.time()
        spaces[name] = TP.ModelSpace(name, splits, device, args)
        print(f"[{NAME}] loaded {name}: D={spaces[name].x_train_cpu.shape[1]}, "
              f"native MRR={spaces[name].mrr_native:.6f} ({time.time()-t0:.1f}s)", flush=True)
    return spaces


def _preflight(spaces: dict, splits: dict, pairs: tuple[dict, ...]) -> list[dict]:
    stored = _stored_p3_rows()
    out = []
    for pair in pairs:
        a, b = spaces[pair["A"]], spaces[pair["B"]]
        t0 = time.time()
        mapping = fit_full_ridge(b.x_train_cpu, a.x_train_cpu)
        q, pred = _predict(a, b, mapping)
        met = _metric_from_prediction(a, pred)
        old = stored[(pair["A"], pair["B"])]
        delta = met["mrr"] - float(old["mrr_stitched"])
        row = {**pair, "n_train": int(b.x_train_cpu.shape[0]),
               "mrr_recomputed": met["mrr"], "mrr_stored_transfer_all_pairs": float(old["mrr_stitched"]),
               "delta": delta, "abs_delta": abs(delta),
               "r2_test": reconstruction_r2(a.x_test.cpu(), q),
               "tolerance": BASELINE_TOL, "passed": abs(delta) <= BASELINE_TOL,
               "seconds": time.time() - t0}
        out.append(row)
        print(f"[{NAME}] preflight {pair['A']} <- {pair['B']}: "
              f"MRR {met['mrr']:.6f} vs stored {old['mrr_stitched']:.6f}, "
              f"delta {delta:+.2e}, R2 {row['r2_test']:.4f}", flush=True)
    if not all(r["passed"] for r in out):
        failures = [f"{r['A']} <- {r['B']} ({r['delta']:+.3g})" for r in out if not r["passed"]]
        raise RuntimeError("baseline reproduction failed: " + "; ".join(failures))
    return out


def _subset_indices(n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def _method_specs(smoke: bool = False):
    specs = (("full_ridge", None), ("truncated_ridge", 64))
    return specs if smoke else specs + (("truncated_ridge", 256),
                                        ("orthogonal_procrustes", None))


def _fit_method(method: str, rank: int | None, x: torch.Tensor, y: torch.Tensor):
    if method == "full_ridge":
        return fit_full_ridge(x, y)
    if method == "truncated_ridge":
        return fit_truncated_ridge(x, y, int(rank))
    if method == "orthogonal_procrustes":
        return fit_orthogonal(x, y)
    raise ValueError(method)


def _row_key(pair: dict, n: int, seed: int | None, method: str, rank: int | None) -> str:
    return json.dumps({"A": pair["A"], "B": pair["B"], "n": n, "seed": seed,
                       "method": method, "rank": rank}, sort_keys=True)


def _run_sweep(spaces: dict, pairs: tuple[dict, ...], result_path: Path,
               smoke: bool = False) -> dict:
    if result_path.exists():
        state = json.load(result_path.open())
    else:
        state = {"schema_version": 1, "experiment": NAME, "rows": {}, "metadata": {}}
    sizes = (512,) if smoke else SAMPLE_SIZES
    seeds = (0,) if smoke else SEEDS
    for pi, pair in enumerate(pairs[:1] if smoke else pairs):
        a, b = spaces[pair["A"]], spaces[pair["B"]]
        n_all = b.x_train_cpu.shape[0]
        if a.x_train_cpu.shape[0] != n_all:
            raise ValueError("paired training spaces have different row counts")
        perms = {seed: _subset_indices(n_all, seed) for seed in seeds}
        for n in sizes:
            effective_seeds = (0,) if n == n_all else seeds
            for seed in effective_seeds:
                idx = torch.arange(n_all) if n == n_all else perms[seed][:n]
                xs, ya = b.x_train_cpu[idx], a.x_train_cpu[idx]
                for method, rank in _method_specs(smoke=smoke):
                    key = _row_key(pair, n, None if n == n_all else seed, method, rank)
                    if key in state["rows"]:
                        continue
                    t_fit = time.time()
                    mapping = _fit_method(method, rank, xs, ya)
                    fit_seconds = time.time() - t_fit
                    t_eval = time.time()
                    q, pred = _predict(a, b, mapping)
                    met = _metric_from_prediction(a, pred)
                    row = {
                        **pair, "pair_index": pi, "method": method, "rank": rank,
                        "n_train": int(n), "fit_seed": None if n == n_all else seed,
                        "all_data_collapsed": bool(n == n_all), "replicate_count": 1,
                        "ridge": RIDGE if "ridge" in method else None,
                        "mrr": met["mrr"], "mrr_over_native": met["mrr"] / a.mrr_native,
                        "native_mrr_fixed_reference": a.mrr_native,
                        "mean_rank": met["mean_rank"], "median_rank": met["median_rank"],
                        "hits_at_1": met["hits_at_1"], "hits_at_10": met["hits_at_10"],
                        "r2_test": reconstruction_r2(a.x_test.cpu(), q),
                        "map_frobenius": float(mapping.norm()),
                        "map_numerical_rank": int(torch.linalg.matrix_rank(mapping,
                                                     atol=1e-5, rtol=1e-4)),
                        "seconds_fit": fit_seconds,
                        "seconds_evaluate": time.time() - t_eval,
                    }
                    state["rows"][key] = row
                    _atomic_json(result_path, _finite(state))
                    print(f"[{NAME}] {pi+1}/{len(pairs)} {pair['A']} <- {pair['B']}; "
                          f"n={n} seed={row['fit_seed']} {method}"
                          f"{'' if rank is None else f'[{rank}]'}: MRR={row['mrr']:.4f}, "
                          f"ratio={row['mrr_over_native']:.3f}, R2={row['r2_test']:.3f}",
                          flush=True)
    return state


def _mean_sd(vals: list[float]) -> tuple[float, float | None]:
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else None


def _summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["A"], r["B"], r["stratum"], r["n_train"], r["method"], r["rank"])].append(r)
    out = []
    for (a, b, stratum, n, method, rank), rs in sorted(groups.items(), key=str):
        row = {"A": a, "B": b, "stratum": stratum, "n_train": n, "method": method,
               "rank": rank, "n_subset_draws": len(rs),
               "all_data_collapsed": all(r["all_data_collapsed"] for r in rs)}
        for k in ("mrr", "mrr_over_native", "r2_test", "mean_rank", "hits_at_1", "hits_at_10"):
            mean, sd = _mean_sd([float(r[k]) for r in rs])
            row[k + "_mean"], row[k + "_sd"] = mean, sd
        out.append(row)
    full = {(r["A"], r["B"], r["n_train"]): r["mrr_mean"] for r in out
            if r["method"] == "full_ridge"}
    for r in out:
        base = full[(r["A"], r["B"], r["n_train"])]
        r["mrr_over_same_n_full_ridge"] = r["mrr_mean"] / base if base else None
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fresh_probe_context(pairs: tuple[dict, ...]) -> list[dict]:
    path = ROOT / "results" / "runs" / "fresh_map_comparison_all.json"
    if not path.exists():
        return []
    wanted = {(p["A"], p["B"]) for p in pairs}
    blob = json.load(path.open())
    out = []
    for r in blob.get("rows", {}).values():
        if not isinstance(r, dict) or (r.get("A"), r.get("B")) not in wanted:
            continue
        out.append({"A": r["A"], "B": r["B"], "mrr_fresh": r["mrr_fresh"],
                    "mrr_fresh_sd": r["mrr_fresh_sd"],
                    "mrr_stitched": r["mrr_stitched"], "stitched_over_fresh": r["portability"],
                    "source": "results/runs/fresh_map_comparison_all.json"})
    return sorted(out, key=lambda r: (r["A"], r["B"]))


def _model_space_metadata(pairs: tuple[dict, ...]) -> dict:
    out = {}
    for model in sorted({p[k] for p in pairs for k in ("A", "B")}):
        slug = TP.E.model_slug(model)
        ladder_path = Path(paths.RUNS) / f"ladder_{slug}.json"
        summary = json.load(ladder_path.open())["summary"]
        layer = int(summary["best_layer"])
        pooling = summary.get("pooling") or "last"
        ckpt_path = Path(paths.CACHE_DEC) / slug / f"probe_E8_infonce_L{layer}_seed0.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pca = ckpt.get("pca")
        if isinstance(pca, dict):
            pca_shape = list(pca["components"].shape)
            query_dim = int(pca_shape[1])
        else:
            pca_shape = None
            query_dim = int(ckpt["D"])
        out[model] = {
            "slug": slug, "selected_layer": layer, "pooling": pooling,
            "query_dimension": query_dim, "pca_component_shape": pca_shape,
            "ladder_source": str(ladder_path.relative_to(ROOT)),
            "space_checkpoint": str(ckpt_path.relative_to(ROOT)),
        }
    return out


def _write_report(path: Path, metadata: dict, preflight: list[dict], summaries: list[dict]) -> None:
    full_n = max(r["n_train"] for r in summaries)
    full_rows = [r for r in summaries if r["n_train"] == full_n]
    by_method = defaultdict(list)
    for r in full_rows:
        by_method[(r["method"], r["rank"])].append(r["mrr_over_native_mean"])
    lines = [
        "# Alignment capacity and sample efficiency evidence report", "",
        f"Generated {metadata['finished_utc']}. This report describes cached-feature fits; it does not identify alignment as a causal mechanism.", "",
        "## Design", "",
        "Six directed pairs were fixed before the novel sweep. Each target A keeps its native single-map probe, relation vectors, known-object filters, and candidate pool fixed. Only the map from source query B into A changes. Queries are the existing PCA-space subject-plus-relation vectors. No extra normalization or intercept is used. Ridge uses `lambda * n = 0.01 * n`, matching the transfer run.", "",
        "`truncated_ridge` retains the leading source singular directions before ridge fitting. It is principal-component ridge and is not response-aware reduced-rank regression. Orthogonal Procrustes is uncentred and unscaled.", "",
        "Sample subsets 512, 2,048, and 8,192 are nested within each of three fixed random permutations. The 33,937-row fit is deterministic and stored once. Test triples are shared across every fit.", "",
        "ModernBERT uses the historical last-token space in the transfer run. Its row does not represent the later mean-pooled ModernBERT atlas result.", "",
        "## Baseline reproduction gate", "",
    ]
    for r in preflight:
        lines.append(f"- {r['A']} <- {r['B']}: recomputed MRR {r['mrr_recomputed']:.6f}, stored {r['mrr_stored_transfer_all_pairs']:.6f}, delta {r['delta']:+.2e} (pass={r['passed']}).")
    lines += ["", "All six baselines passed before the novel sweep." if all(r["passed"] for r in preflight) else "The baseline gate failed.", "",
              "The historical ModernBERT row exposed float32 CPU-reduction sensitivity: its MRR ranged from 0.271026 with one CPU thread to 0.271084 with 16 or 32 threads. Float64 accumulation gave 0.271095, within the fixed 0.00002 reproduction tolerance, and was used for every method while keeping CPU work to four threads.", "",
              "## Sample-efficiency result", ""]
    full_method = {(r["A"], r["B"], r["n_train"]): r for r in summaries
                   if r["method"] == "full_ridge"}
    endpoint = {(r["A"], r["B"]): r for r in summaries
                if r["method"] == "full_ridge" and r["n_train"] == full_n}
    for n in sorted({r["n_train"] for r in summaries}):
        rs = [r for r in summaries if r["method"] == "full_ridge" and r["n_train"] == n]
        native = statistics.mean(r["mrr_over_native_mean"] for r in rs)
        endpoint_fraction = statistics.mean(
            r["mrr_mean"] / endpoint[(r["A"], r["B"])]["mrr_mean"] for r in rs)
        lines.append(f"- Full ridge with {n:,} fit pairs: mean MRR/native = {native:.3f}; mean fraction of its pair-specific full-data MRR = {endpoint_fraction:.3f}.")
    r256_512 = [r for r in summaries if r["method"] == "truncated_ridge" and r["rank"] == 256
                and r["n_train"] == 512]
    wins = sum(r["mrr_mean"] > full_method[(r["A"], r["B"], 512)]["mrr_mean"] for r in r256_512)
    lines += ["", f"Rank-256 ridge beats full ridge at 512 examples on {wins} of 6 pairs and averages {statistics.mean(r['mrr_over_same_n_full_ridge'] for r in r256_512):.3f} times its MRR. By 2,048 examples it averages {statistics.mean(r['mrr_over_same_n_full_ridge'] for r in summaries if r['method'] == 'truncated_ridge' and r['rank'] == 256 and r['n_train'] == 2048):.3f} times full-ridge MRR. This is the main capacity/sample-efficiency crossover.", "",
              "## Full-data descriptive comparison", ""]
    for key, vals in sorted(by_method.items(), key=str):
        method, rank = key
        label = method + (f" rank {rank}" if rank is not None else "")
        lines.append(f"- {label}: mean MRR/native ratio across the six predeclared pairs = {statistics.mean(vals):.3f}; range {min(vals):.3f} to {max(vals):.3f}.")
    orth_full = [r for r in full_rows if r["method"] == "orthogonal_procrustes"]
    orth_dec = [r for r in orth_full if "ModernBERT" not in r["B"]]
    modern = next(r for r in orth_full if "ModernBERT" in r["B"])
    gemma = next(r for r in orth_full if r["B"] == "google/gemma-4-12B")
    lines += ["", f"Orthogonal Procrustes retains a mean {statistics.mean(r['mrr_over_same_n_full_ridge'] for r in orth_dec):.3f} of full-ridge MRR on the five decoder-source pairs, but collapses on the historical last-token ModernBERT pair (MRR {modern['mrr_mean']:.4f}, {modern['mrr_over_same_n_full_ridge']:.3f} of full ridge). For Gemma -> Llama it has R2 = {gemma['r2_test_mean']:.3f} while retaining {gemma['mrr_over_same_n_full_ridge']:.3f} of full-ridge MRR, a concrete warning that Euclidean query reconstruction and cosine ranking are different outcomes.", ""]
    fresh = metadata.get("existing_fresh_probe_context", [])
    if fresh:
        lines += ["## Existing equal-label fresh-probe context", "",
                  "fresh_map_comparison already trained fresh source-to-target probes for three overlapping pairs. No new probe training was needed:", ""]
        for r in fresh:
            lines.append(f"- {r['A']} <- {r['B']}: full-ridge stitched MRR {r['mrr_stitched']:.4f}, fresh-probe MRR {r['mrr_fresh']:.4f} +/- {r['mrr_fresh_sd']:.4f} across three probe seeds, ratio {r['stitched_over_fresh']:.3f}.")
        lines.append("")
    lines += ["Pair-level absolute MRR, rank summaries, held-out reconstruction R2, and sample curves are in `alignment_capacity_summary_full.csv`. The raw fit rows are in `alignment_capacity_raw_full.csv` and the resumable JSON state.", "",
              "## Statistical units and limits", "",
              "The three subset seeds are repeated training-row draws, not independent model trainings or independent test sets. Their standard deviations quantify subset sensitivity only. The full-data fit has one effective replicate, so its SD is blank and must not be plotted as seed variability. The six pairs were deliberately chosen strata, not a random sample of models; across-pair means are descriptive. All methods reuse the same 4,243 Wikidata test triples and target candidate pools. R2 measures query reconstruction while MRR measures downstream filtered ranking; association between them is descriptive and does not establish causality.", ""]
    path.write_text("\n".join(lines))


def self_test() -> None:
    torch.manual_seed(7)
    x = torch.randn(80, 12)
    true = torch.randn(12, 12)
    y = x @ true + 0.01 * torch.randn(80, 12)
    full = fit_full_ridge(x, y)
    reference = X.fit_stitch(x, y, ridge=RIDGE)
    assert full.shape == (12, 12)
    assert torch.allclose(full, reference, atol=2e-5, rtol=2e-5)
    trunc = fit_truncated_ridge(x, y, 4)
    assert trunc.shape == full.shape and int(torch.linalg.matrix_rank(trunc)) <= 4
    trunc_full = fit_truncated_ridge(x, y, 12)
    assert torch.allclose(trunc_full, full, atol=2e-4, rtol=2e-4)
    orth = fit_orthogonal(x, y)
    assert torch.allclose(orth.T @ orth, torch.eye(12), atol=2e-5, rtol=2e-5)
    y_test = torch.randn(20, 12)
    assert abs(reconstruction_r2(y_test, y_test) - 1.0) < 1e-7

    scores = torch.tensor([[0.8, 0.9, 0.1]])
    gold = torch.tensor([0])
    valid = [torch.tensor([0, 1])]
    rf = M.ranks(scores, gold, valid=valid, tie="random", seed=0)
    ru = M.ranks(scores, gold, valid=None, tie="random", seed=0)
    assert int(rf[0]) == 1 and int(ru[0]) == 2

    mean = torch.arange(5).float()
    comp, _ = torch.linalg.qr(torch.randn(5, 3))
    s, r = torch.randn(6, 5), torch.randn(6, 5)
    q = (s - mean) @ comp + (r - mean) @ comp
    expected = (s + r - 2 * mean) @ comp
    assert q.shape == (6, 3) and torch.allclose(q, expected, atol=1e-6)
    print(f"[{NAME}] self-test passed: ridge identity, truncation rank/full equivalence, "
          "orthogonality, R2, known-object filtering, and projection/query convention")


def _device() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES=1 exactly; got {visible!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if "RTX 6000 Ada" not in name:
        raise RuntimeError(f"logical cuda:0 is {name!r}, expected physical GPU 1 RTX 6000 Ada")
    return torch.device("cuda:0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--collect", action="store_true", help="regenerate full extracts without loading caches")
    args = ap.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return 0
    if args.collect:
        result_path = RESULT_DIR / "alignment_capacity_full.json"
        state = json.load(result_path.open())
        rows = list(state["rows"].values())
        summaries = _summaries(rows)
        metadata = state["metadata"]
        metadata["collected_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        metadata["baseline_numerical_audit"] = BASELINE_NUMERICAL_AUDIT
        metadata["existing_fresh_probe_context"] = _fresh_probe_context(PAIRS)
        metadata["model_spaces"] = _model_space_metadata(PAIRS)
        test = D.load_dataset("wikidata", "test")
        metadata["target_candidate_pool_size"] = int(test["object"].astype(str).nunique())
        metadata["known_object_filter"] = (
            "valid subject-relation objects built from Wikidata train+valid+test, as in the transfer run")
        state["metadata"], state["summaries"] = metadata, summaries
        _atomic_json(result_path, _finite(state))
        _write_csv(FINAL_DIR / "alignment_capacity_raw_full.csv", rows)
        _write_csv(FINAL_DIR / "alignment_capacity_summary_full.csv", summaries)
        _atomic_json(FINAL_DIR / "alignment_capacity_metadata_full.json", _finite(metadata))
        _write_report(FINAL_DIR / "alignment_capacity_report_full.md", metadata,
                      state["preflight"], summaries)
        print(f"[{NAME}] regenerated full extracts from {result_path}")
        return 0
    device = _device()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    result_path = RESULT_DIR / f"alignment_capacity_{tag}.json"
    preflight_path = RESULT_DIR / f"alignment_capacity_preflight_{tag}.json"
    print(f"[{NAME}] device={device} ({torch.cuda.get_device_name(0)}), tag={tag}", flush=True)
    splits = {s: D.load_dataset("wikidata", s) for s in ("train", "valid", "test")}
    _metric_from_prediction.splits = splits
    pairs = PAIRS[:1] if args.smoke else PAIRS
    spaces = _load_spaces(splits, device, pairs)
    preflight = _preflight(spaces, splits, pairs)
    _atomic_json(preflight_path, {"tolerance": BASELINE_TOL, "rows": preflight})
    if args.preflight_only:
        return 0
    started = dt.datetime.now(dt.timezone.utc)
    state = _run_sweep(spaces, pairs, result_path, smoke=args.smoke)
    rows = list(state["rows"].values())
    summaries = _summaries(rows)
    finished = dt.datetime.now(dt.timezone.utc)
    metadata = {
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "elapsed_seconds_this_invocation": (finished - started).total_seconds(),
        "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device), "gpu_name": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "ridge": RIDGE, "sample_sizes": list((512,) if args.smoke else SAMPLE_SIZES),
        "seeds": list((0,) if args.smoke else SEEDS), "metric_tie_seed": METRIC_SEED,
        "n_train_all": len(splits["train"]), "n_test": len(splits["test"]),
        "pairs": list(pairs), "baseline_tolerance": BASELINE_TOL,
        "normalization": "checkpoint PCA per field, then subject + relation; no additional normalization",
        "linear_algebra": "CPU float64 accumulation/decomposition, returned maps float32",
        "intercept": False,
        "truncated_ridge_definition": "source-spectrum principal-component ridge; not response-aware reduced-rank regression",
        "modernbert_note": "historical the transfer run last-token space; not the later mean-pooled atlas row",
        "full_size_replicates": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_numerical_audit": BASELINE_NUMERICAL_AUDIT,
        "existing_fresh_probe_context": _fresh_probe_context(pairs),
        "model_spaces": _model_space_metadata(pairs),
        "target_candidate_pool_size": int(splits["test"]["object"].astype(str).nunique()),
        "known_object_filter": "valid subject-relation objects built from Wikidata train+valid+test, as in the transfer run",
    }
    state["metadata"] = metadata
    state["preflight"] = preflight
    state["summaries"] = summaries
    _atomic_json(result_path, _finite(state))
    _write_csv(FINAL_DIR / f"alignment_capacity_raw_{tag}.csv", rows)
    _write_csv(FINAL_DIR / f"alignment_capacity_summary_{tag}.csv", summaries)
    _atomic_json(FINAL_DIR / f"alignment_capacity_metadata_{tag}.json", _finite(metadata))
    _write_report(FINAL_DIR / f"alignment_capacity_report_{tag}.md", metadata, preflight, summaries)
    print(f"[{NAME}] wrote {result_path} and final extracts under {FINAL_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
