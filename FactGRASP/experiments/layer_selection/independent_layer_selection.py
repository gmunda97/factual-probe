# table 7 and figure 5
# each of the 4 configs picks its own layer out of 5 fixed layers, on bert wikidata, bert conceptnet,
# qwen3.5-4b and gemma-4-e4b wikidata. selection trains one map (seed 1729) on 8000 train rows and uses
# valid mrr, then the selected and the fixed layer are trained on the full split with seeds 0 1 2
# run: python -m experiments.layer_selection.independent_layer_selection --stage all
# figure: python plots/make_layer_selection_figure.py

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("FACTGRASP_FORCE_GPU", "0")

import pandas as pd
import torch

from factgrasp import data as D
from factgrasp import encoders as E
from factgrasp import gpu_guard
from factgrasp import metrics as M
from factgrasp import paths

sys.path.insert(0, paths.KG_CODE)
import common as C
import variants as V


NAME = "independent_layer_selection"
RESULT_ROOT = Path(paths.ROOT) / "results" / "appendix" / NAME
FINAL_ROOT = Path(paths.ROOT) / "final_results" / "appendix"
FEATURE_ROOT = RESULT_ROOT / "feature_cache"
CELL_ROOT = RESULT_ROOT / "cells"
PCA_ROOT = RESULT_ROOT / "pca"
STATE_PATH = RESULT_ROOT / "state.json"
DESIGN_PATH = RESULT_ROOT / "design.json"
FIELDS = ("subject", "pred_value", "object")


@dataclass(frozen=True)
class PairSpec:
    key: str
    model: str
    dataset: str
    pooling: str
    layers: tuple[int, ...]
    shared_layer: int


PAIRS = (
    PairSpec("bert_wikidata", "bert-base-uncased", "wikidata", "mean",
             (2, 4, 6, 10, 12), 10),
    PairSpec("bert_conceptnet", "bert-base-uncased", "conceptnet", "mean",
             (2, 4, 6, 10, 12), 10),
    PairSpec("qwen4b_wikidata", "Qwen/Qwen3.5-4B", "wikidata", "last",
             (6, 12, 20, 26, 32), 26),
    PairSpec("gemmae4b_wikidata", "google/gemma-4-E4B", "wikidata", "last",
             (9, 18, 24, 33, 42), 33),
)

RECIPES = {
    "single_cosine": (1, "cosine"),
    "mixture8_cosine": (8, "cosine"),
    "single_composite_infonce": (1, "infonce"),
    "mixture8_composite_infonce": (8, "infonce"),
}

DEFAULTS = {
    "selection_seed": 1729,
    "final_seeds": [0, 1, 2],
    "calibration_rows": 8000,
    "pca": 768,
    "experts": 8,
    "epochs": 60,
    "patience": 12,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "temperature": 0.05,
    "balance_penalty": 0.01,
    "tie": "random",
    "pca_sample": 20000,
    "pca_niter": 4,
}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as fh:
            return json.load(fh)
    return {"selection": {}, "selected_layers": {}, "final": {}, "events": []}


def save_state(state: dict) -> None:
    atomic_json(STATE_PATH, state)


def event(state: dict, kind: str, **fields) -> None:
    state.setdefault("events", []).append({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime()),
                                             "kind": kind, **fields})
    save_state(state)


def frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in df.loc[:, list(FIELDS)].astype(str).itertuples(index=False, name=None):
        h.update("\x1f".join(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_design(args) -> None:
    design = {
        "name": NAME,
        "frozen_before_test": True,
        "pairs": [asdict(p) for p in PAIRS],
        "recipes": {k: {"experts": e, "loss": loss} for k, (e, loss) in RECIPES.items()},
        "selection": "maximum filtered validation MRR; ties choose the shallower layer",
        "comparison": "same new implementation and seeds at independently selected and fixed shared layers",
        "test_access": "test embeddings/pools are loaded only in the final stage, after selection is saved",
        "scope": "bounded three-model, four-model-graph-cell pilot; no population-general claim",
        "coarse_grid_limitation": "five predeclared depths can miss a better intervening layer",
        "smollm": "excluded before outcomes, by runtime-bounded pilot decision",
        "hyperparameters": {**DEFAULTS, "epochs": args.epochs, "patience": args.patience,
                            "batch_size": args.batch_size},
        "bert_conceptnet_cache": "missing multi-layer representations are generated once into the isolated revision feature cache; legacy cache is read-only",
    }
    if DESIGN_PATH.exists():
        old = json.load(open(DESIGN_PATH))
        serialised = json.loads(json.dumps(design, sort_keys=True))
        if old != serialised:
            raise RuntimeError("frozen design differs from existing design.json; use the same settings")
    else:
        atomic_json(DESIGN_PATH, design)


def assert_gpu_zero(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("this real experiment requires CUDA")
    name = torch.cuda.get_device_name(device)
    if "A6000" not in name or "Ada" in name:
        raise RuntimeError(f"physical GPU 0 must be the RTX A6000; torch sees {name!r}")


def check_gpu_contention() -> None:
    if gpu_guard.check_yield():
        gpu_guard.yield_and_exit("independent-layer state is checkpointed")


def revision_feature_path(spec: PairSpec, split: str, field: str, layer: int) -> Path:
    slug = E.model_slug(spec.model)
    return FEATURE_ROOT / slug / f"{spec.dataset}_{split}_{field}_{spec.pooling}_L{layer}.pt"


def _valid_tensor(path: Path, n: int) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(path, map_location="cpu")
    if t.ndim != 2 or int(t.shape[0]) != int(n):
        return None
    return t


def load_fields(spec: PairSpec, split: str, df: pd.DataFrame, layers: list[int],
                device: torch.device, model_holder: list, state: dict,
                batch: int = 256) -> dict:
    out = {f: {} for f in FIELDS}
    for field in FIELDS:
        missing = []
        for layer in layers:
            legacy = Path(E.cache_path(spec.model, spec.dataset, split, field,
                                       spec.pooling, layer))
            t = _valid_tensor(legacy, len(df))
            source = "legacy"
            if t is None:
                rev = revision_feature_path(spec, split, field, layer)
                t = _valid_tensor(rev, len(df))
                source = "revision"
            if t is None:
                missing.append(layer)
            else:
                out[field][layer] = t
                state.setdefault("feature_sources", {})[
                    f"{spec.key}|{split}|{field}|L{layer}"] = source
        if missing:
            if spec.model != "bert-base-uncased":
                raise FileNotFoundError(
                    f"missing legacy representation for {spec.key} {split}/{field} {missing}; "
                    "large-model forward regeneration is disabled")
            if model_holder[0] is None:
                print(f"[{NAME}] loading BERT once for isolated missing-feature generation",
                      flush=True)
                model_holder[:] = list(E.load_decoder(spec.model, device, dtype="fp32",
                                                       prefer="auto"))
            fresh = E.embed_strings(df[field].astype(str).tolist(), model_holder[0],
                                    model_holder[1], missing, spec.pooling, batch=batch,
                                    max_len=32, device=device, frame="final")
            for layer, tensor in fresh.items():
                path = revision_feature_path(spec, split, field, layer)
                atomic_torch(path, tensor)
                out[field][layer] = tensor
                state.setdefault("feature_sources", {})[
                    f"{spec.key}|{split}|{field}|L{layer}"] = "revision_generated"
                event(state, "feature_generated", pair=spec.key, split=split,
                      field=field, layer=layer, rows=len(df), path=str(path))
    save_state(state)
    for field in FIELDS:
        for layer in layers:
            if out[field][layer].shape[0] != len(df):
                raise AssertionError("representation/data row count mismatch")
    return out


def pca_path(spec: PairSpec, stage: str, layer: int) -> Path:
    return PCA_ROOT / f"{spec.key}_{stage}_L{layer}.pt"


def fit_or_load_pca(spec: PairSpec, stage: str, layer: int, train_fields: dict,
                    device: torch.device, pca_dim: int) -> Optional[E.PCAProjector]:
    d = int(train_fields["subject"][layer].shape[1])
    if d <= pca_dim:
        return None
    path = pca_path(spec, stage, layer)
    if path.exists():
        return E.PCAProjector.load(str(path))
    rows = torch.cat([train_fields[f][layer].float() for f in FIELDS], dim=0)
    gen = torch.Generator().manual_seed(0)
    if rows.shape[0] > DEFAULTS["pca_sample"]:
        idx = torch.randperm(rows.shape[0], generator=gen)[:DEFAULTS["pca_sample"]]
        rows = rows[idx]
    sample = rows.to(device)
    mean = sample.mean(0)
    q = min(pca_dim, sample.shape[0], sample.shape[1])
    torch.manual_seed(0)
    _, _, components = torch.pca_lowrank(sample - mean, q=q, center=False,
                                         niter=DEFAULTS["pca_niter"])
    proj = E.PCAProjector(mean.cpu(), components[:, :q].contiguous().cpu())
    atomic_torch(path, proj.state_dict())
    del rows, sample, components
    torch.cuda.empty_cache()
    return proj


def transform(t: torch.Tensor, proj: Optional[E.PCAProjector],
              device: torch.device, chunk: int = 8192) -> torch.Tensor:
    if proj is None:
        return t.float()
    mean = proj.mean.to(device)
    comp = proj.components.to(device)
    out = []
    for i in range(0, len(t), chunk):
        out.append(((t[i:i + chunk].float().to(device) - mean) @ comp).cpu())
    return torch.cat(out, 0)


def project_layer(fields_by_split: dict, spec: PairSpec, stage: str, layer: int,
                  device: torch.device, pca_dim: int) -> dict:
    proj = fit_or_load_pca(spec, stage, layer, fields_by_split["train"], device, pca_dim)
    out = {}
    for split, fs in fields_by_split.items():
        z = {f: transform(fs[f][layer], proj, device) for f in FIELDS}
        out[split] = {"x": z["subject"] + z["pred_value"],
                      "relation": z["pred_value"], "object": z["object"]}
    return out


def pool_and_gold(df: pd.DataFrame, objects: torch.Tensor, valid_map: dict):
    names, pool, gold = C.build_unique_object_pool(df, objects)
    valid_lists = D.valid_index_lists(df, names, valid_map)
    if len(gold) != len(df) or len(valid_lists) != len(df):
        raise AssertionError("metric inputs are not row aligned")
    for i, name in enumerate(df["object"].astype(str)):
        if names[int(gold[i])] != name:
            raise AssertionError(f"gold pool mapping mismatch at row {i}")
    return names, pool, gold, valid_lists


def clone_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_probe(tensors: dict, recipe: str, seed: int, device: torch.device,
                args) -> tuple[torch.nn.Module, list[dict]]:
    experts, loss_name = RECIPES[recipe]
    C.set_seed(seed)
    torch.manual_seed(seed)
    tr, va = tensors["train"], tensors["valid"]
    model = V.FlexMixtureOfW(tr["x"].shape[1], tr["object"].shape[1], experts,
                             gate_dim=tr["relation"].shape[1],
                             gate_type="softmax").to(device)
    loss_fn = V.make_loss(loss_name, temp=DEFAULTS["temperature"])
    opt = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["lr"],
                            weight_decay=DEFAULTS["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=4)
    xt, rt, ot = (tr[k].to(device) for k in ("x", "relation", "object"))
    xv, rv, ov = (va[k].to(device) for k in ("x", "relation", "object"))
    best = float("inf")
    best_epoch = -1
    best_state = clone_state(model)
    bad = 0
    curve = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        seen = 0
        for i in range(0, len(xt), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pred, gate = model(xt[idx], rt[idx])
            loss = loss_fn(pred, ot[idx])
            if experts > 1:
                loss = loss + DEFAULTS["balance_penalty"] * V.load_balance_loss(gate)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            pred_v, _ = model(xv, rv)
            val_loss = float(loss_fn(pred_v, ov).item())
        sched.step(val_loss)
        curve.append({"epoch": epoch + 1, "train_loss": total / max(seen, 1),
                      "val_loss": val_loss, "lr": float(opt.param_groups[0]["lr"])})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch + 1
            best_state = clone_state(model)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    for row in curve:
        row["selected_epoch"] = (row["epoch"] == best_epoch)
    return model, curve


@torch.no_grad()
def predict(model, tensors: dict, split: str, device: torch.device,
            chunk: int = 8192) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    x, r = tensors[split]["x"], tensors[split]["relation"]
    preds, gates = [], []
    for i in range(0, len(x), chunk):
        pred, gate = model(x[i:i + chunk].to(device), r[i:i + chunk].to(device))
        preds.append(pred.float().cpu())
        if gate is not None:
            gates.append(gate.float().cpu())
    return torch.cat(preds), torch.cat(gates) if gates else None


def json_metrics(metrics: dict) -> dict:
    return M.jsonable(metrics)


def cell_path(spec: PairSpec, stage: str, recipe: str, layer: int, seed: int) -> Path:
    return CELL_ROOT / spec.key / f"{stage}_{recipe}_L{layer}_seed{seed}.pt"


def run_cell(spec: PairSpec, stage: str, recipe: str, layer: int, seed: int,
             tensors: dict, eval_df: pd.DataFrame, valid_map: dict,
             device: torch.device, args) -> dict:
    path = cell_path(spec, stage, recipe, layer, seed)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    t0 = time.time()
    model, curve = train_probe(tensors, recipe, seed, device, args)
    split = "valid" if stage in ("selection", "smoke_selection") else "test"
    names, pool, gold, lists = pool_and_gold(eval_df, tensors[split]["object"], valid_map)
    pred, gates = predict(model, tensors, split, device)
    measured = M.evaluate(pred, pool, gold, valid=lists, tie=DEFAULTS["tie"],
                          tau=DEFAULTS["temperature"], seed=seed)
    row = {
        "pair": spec.key, "model": spec.model, "dataset": spec.dataset,
        "stage": stage, "recipe": recipe, "layer": layer, "seed": seed,
        "metrics": json_metrics(measured), "rr": measured["rr"].cpu(),
        "ranks": measured["ranks"].cpu(), "curve": curve,
        "seconds": time.time() - t0, "pool_names": names,
        "gate_mean": gates.mean(0) if gates is not None else None,
        "epochs_run": len(curve),
        "best_epoch": next(r["epoch"] for r in curve if r["selected_epoch"]),
    }
    atomic_torch(path, row)
    del model, pred, gates
    torch.cuda.empty_cache()
    print(f"[{stage}] {spec.key} {recipe} L{layer} seed={seed} "
          f"MRR={measured['MRR']:.4f} epochs={row['epochs_run']} "
          f"seconds={row['seconds']:.1f}", flush=True)
    return row


def load_pair_frames(spec: PairSpec) -> dict:
    return {s: D.load_dataset(spec.dataset, s).reset_index(drop=True)
            for s in ("train", "valid", "test")}


def valid_map_all(frames: dict) -> dict:
    return D.valid_sets(pd.concat([frames[s] for s in ("train", "valid", "test")],
                                  ignore_index=True))


def selection_for_pair(spec: PairSpec, frames: dict, state: dict,
                       model_holder: list, device: torch.device, args,
                       smoke: bool = False) -> None:
    layers = list(spec.layers if not smoke else (spec.layers[0], spec.shared_layer))
    ncal = min(DEFAULTS["calibration_rows"], len(frames["train"]))
    calib = frames["train"].sample(n=ncal, random_state=0).reset_index(drop=True)
    valid = frames["valid"]
    if smoke:
        calib = calib.head(384).reset_index(drop=True)
        valid = valid.head(384).reset_index(drop=True)
    full_calib = frames["train"].sample(n=ncal, random_state=0).reset_index(drop=True)
    tr_all = load_fields(spec, "traincal8000", full_calib, layers, device,
                         model_holder, state)
    va_all = load_fields(spec, "valid", frames["valid"], layers, device,
                         model_holder, state)
    if smoke:
        tr_all = {f: {l: t[:len(calib)] for l, t in by.items()} for f, by in tr_all.items()}
        va_all = {f: {l: t[:len(valid)] for l, t in by.items()} for f, by in va_all.items()}
    vm = valid_map_all(frames)
    stage_name = "smoke_selection" if smoke else "selection"
    seed = DEFAULTS["selection_seed"]
    for layer in layers:
        projected = project_layer({"train": tr_all, "valid": va_all}, spec,
                                  stage_name, layer, device, args.pca)
        for recipe in RECIPES:
            row = run_cell(spec, stage_name, recipe, layer, seed, projected,
                           valid, vm, device, args)
            if not smoke:
                key = f"{recipe}|L{layer}"
                state["selection"].setdefault(spec.key, {})[key] = {
                    "MRR": row["metrics"]["MRR"], "Hits@1": row["metrics"]["Hits@1"],
                    "seconds": row["seconds"], "epochs_run": row["epochs_run"],
                    "best_epoch": row["best_epoch"], "cell": str(cell_path(
                        spec, "selection", recipe, layer, seed)),
                }
                save_state(state)
            check_gpu_contention()
        del projected
    if smoke:
        return
    selected = {}
    for recipe in RECIPES:
        points = [(state["selection"][spec.key][f"{recipe}|L{l}"]["MRR"], l)
                  for l in spec.layers]
        selected[recipe] = min(points, key=lambda x: (-x[0], x[1]))[1]
    state["selected_layers"][spec.key] = selected
    event(state, "selection_locked", pair=spec.key, layers=selected)
    print(f"[selection] locked {spec.key}: {selected}", flush=True)


def final_for_pair(spec: PairSpec, frames: dict, state: dict, model_holder: list,
                   device: torch.device, args) -> None:
    if spec.key not in state.get("selected_layers", {}):
        raise RuntimeError(f"selection for {spec.key} is not locked")
    selected = state["selected_layers"][spec.key]
    layers = sorted(set(selected.values()) | {spec.shared_layer})
    fields = {
        "train": load_fields(spec, "train", frames["train"], layers, device,
                             model_holder, state),
        "valid": load_fields(spec, "valid", frames["valid"], layers, device,
                             model_holder, state),
        "test": load_fields(spec, "test", frames["test"], layers, device,
                            model_holder, state),
    }
    vm = valid_map_all(frames)
    for layer in layers:
        projected = project_layer(fields, spec, "final_fulltrain", layer, device, args.pca)
        for recipe in RECIPES:
            roles = []
            if layer == selected[recipe]:
                roles.append("independent")
            if layer == spec.shared_layer:
                roles.append("fixed_shared")
            if not roles:
                continue
            for seed in DEFAULTS["final_seeds"]:
                row = run_cell(spec, "final", recipe, layer, seed, projected,
                               frames["test"], vm, device, args)
                key = f"{recipe}|L{layer}|seed{seed}"
                state["final"].setdefault(spec.key, {})[key] = {
                    "roles": roles, "MRR": row["metrics"]["MRR"],
                    "Hits@1": row["metrics"]["Hits@1"], "seconds": row["seconds"],
                    "epochs_run": row["epochs_run"], "best_epoch": row["best_epoch"],
                    "cell": str(cell_path(spec, "final", recipe, layer, seed)),
                }
                save_state(state)
                check_gpu_contention()
        del projected


def ci_dict(a: torch.Tensor, b: torch.Tensor) -> dict:
    return M.paired_ci(a.float(), b.float(), n_boot=5000, seed=0)


def final_rows(state: dict) -> tuple[list[dict], dict]:
    rows = []
    loaded = {}
    for spec in PAIRS:
        selected = state["selected_layers"][spec.key]
        for recipe in RECIPES:
            for role, layer in (("independent", selected[recipe]),
                                ("fixed_shared", spec.shared_layer)):
                cells = []
                for seed in DEFAULTS["final_seeds"]:
                    key = f"{recipe}|L{layer}|seed{seed}"
                    path = Path(state["final"][spec.key][key]["cell"])
                    cells.append(torch.load(path, map_location="cpu", weights_only=False))
                agg = M.aggregate_seeds([c["metrics"] for c in cells])
                mean_rr = torch.stack([c["rr"].float() for c in cells]).mean(0)
                loaded[(spec.key, role, recipe)] = mean_rr
                rows.append({"pair": spec.key, "model": spec.model,
                             "dataset": spec.dataset, "role": role,
                             "recipe": recipe, "layer": layer,
                             "MRR_mean": agg["MRR"], "MRR_sd": agg["MRR_std"],
                             "MRR_ci95": agg["MRR_ci95"],
                             "Hits1_mean": agg["Hits@1"],
                             "mean_epochs": sum(c["epochs_run"] for c in cells) / len(cells),
                             "mean_seconds": sum(c["seconds"] for c in cells) / len(cells)})
    return rows, loaded


def build_report(state: dict) -> None:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    selection_csv = FINAL_ROOT / "independent_layer_selection_validation_curves.csv"
    with open(selection_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "recipe", "layer", "MRR", "Hits@1",
                                               "epochs_run", "best_epoch", "seconds"])
        w.writeheader()
        for spec in PAIRS:
            for recipe in RECIPES:
                for layer in spec.layers:
                    r = state["selection"][spec.key][f"{recipe}|L{layer}"]
                    w.writerow({"pair": spec.key, "recipe": recipe, "layer": layer, **r,
                                "cell": None} if False else {
                                    "pair": spec.key, "recipe": recipe, "layer": layer,
                                    **{k: r[k] for k in ("MRR", "Hits@1", "epochs_run",
                                                         "best_epoch", "seconds")}})
    rows, rr = final_rows(state)
    test_csv = FINAL_ROOT / "independent_layer_selection_test_metrics.csv"
    with open(test_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    comparisons = {}
    for spec in PAIRS:
        comparisons[spec.key] = {"mixture_advantage": {}, "layer_selection_gain": {}}
        for role in ("independent", "fixed_shared"):
            gains = {}
            for obj in ("cosine", "composite_infonce"):
                mix = rr[(spec.key, role, f"mixture8_{obj}")]
                single = rr[(spec.key, role, f"single_{obj}")]
                gains[obj] = ci_dict(mix, single)
            gains["composite_minus_cosine"] = ci_dict(
                rr[(spec.key, role, "mixture8_composite_infonce")]
                - rr[(spec.key, role, "single_composite_infonce")],
                rr[(spec.key, role, "mixture8_cosine")]
                - rr[(spec.key, role, "single_cosine")])
            comparisons[spec.key]["mixture_advantage"][role] = gains
        for recipe in RECIPES:
            comparisons[spec.key]["layer_selection_gain"][recipe] = ci_dict(
                rr[(spec.key, "independent", recipe)],
                rr[(spec.key, "fixed_shared", recipe)])
    comp_path = FINAL_ROOT / "independent_layer_selection_paired_differences.json"
    atomic_json(comp_path, comparisons)

    lines = [
        "# Independent layer selection for FactGRASP recipes", "",
        "This is a bounded three-model, four-model-graph-cell experiment. It does not support a population-general claim. Each recipe selected among five predeclared coarse depths using validation MRR; an intervening layer could be better.", "",
        "## Selected layers", "",
        "| Pair | Fixed layer | Single cosine | Mixture cosine | Single composite InfoNCE | Mixture composite InfoNCE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in PAIRS:
        s = state["selected_layers"][spec.key]
        lines.append(f"| {spec.key} | {spec.shared_layer} | {s['single_cosine']} | "
                     f"{s['mixture8_cosine']} | {s['single_composite_infonce']} | "
                     f"{s['mixture8_composite_infonce']} |")
    lines += ["", "## Test MRR", "",
              "Mean and 95% t interval half-width over the three training seeds.", "",
              "| Pair | Layer rule | Recipe | Layer | MRR |",
              "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['pair']} | {row['role']} | {row['recipe']} | {row['layer']} | "
                     f"{row['MRR_mean']:.4f} ± {row['MRR_ci95']:.4f} |")
    lines += ["", "## Paired test-triple differences", "",
              "Intervals below are paired bootstrap 95% intervals over test triples after averaging reciprocal rank over seeds. `composite_minus_cosine` is the composite InfoNCE mixture-minus-single gain minus the cosine mixture-minus-single gain.", ""]
    for spec in PAIRS:
        lines.append(f"### {spec.key}")
        lines.append("")
        lines.append("| Layer rule | Cosine mixture gain | Composite mixture gain | Composite minus cosine |")
        lines.append("|---|---:|---:|---:|")
        for role in ("fixed_shared", "independent"):
            g = comparisons[spec.key]["mixture_advantage"][role]
            def fmt(x):
                return (f"{x['diff']:+.4f} "
                        f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]")
            lines.append(f"| {role} | {fmt(g['cosine'])} | "
                         f"{fmt(g['composite_infonce'])} | "
                         f"{fmt(g['composite_minus_cosine'])} |")
        lines.append("")
    lines += ["## Protocol and artifacts", "",
              "The fixed-layer reference was retrained with the same code, seed set, early-stopping budget, PCA rule, filtering, and candidate pool as the independently selected arms. Cosine and composite InfoNCE use the original gate load-balancing penalty of 0.01 for the mixture. BERT ConceptNet representations absent from the legacy cache were generated into the isolated revision cache; the legacy cache was not changed.", "",
              f"Raw cell tensors (including reciprocal ranks and epoch loss curves): `{CELL_ROOT}`  ",
              f"Validation curves: `{selection_csv}`  ",
              f"Test aggregates: `{test_csv}`  ",
              f"Paired differences: `{comp_path}`", ""]
    report_path = FINAL_ROOT / "independent_layer_selection.md"
    report_path.write_text("\n".join(lines))
    atomic_json(FINAL_ROOT / "independent_layer_selection_summary.json",
                {"selected_layers": state["selected_layers"], "test_rows": rows,
                 "paired_differences": comparisons})
    print(f"[report] wrote {report_path}", flush=True)


def all_complete(state: dict) -> bool:
    for spec in PAIRS:
        if spec.key not in state.get("selected_layers", {}):
            return False
        selected = state["selected_layers"][spec.key]
        for recipe in RECIPES:
            for layer in {selected[recipe], spec.shared_layer}:
                for seed in DEFAULTS["final_seeds"]:
                    if f"{recipe}|L{layer}|seed{seed}" not in state.get("final", {}).get(spec.key, {}):
                        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="independent validation layer selection for the four configs")
    ap.add_argument("--stage", choices=("selection", "final", "all", "report"),
                    default="all")
    ap.add_argument("--pairs", default=",".join(p.key for p in PAIRS))
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--pca", type=int, default=DEFAULTS["pca"])
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs = 2
        args.patience = 2
        args.pairs = "bert_wikidata"
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELL_ROOT.mkdir(parents=True, exist_ok=True)
    write_design(args) if not args.smoke else None
    state = load_state()
    selected_keys = [x.strip() for x in args.pairs.split(",") if x.strip()]
    unknown = set(selected_keys) - {p.key for p in PAIRS}
    if unknown:
        raise ValueError(f"unknown pair keys: {sorted(unknown)}")
    specs = [p for p in PAIRS if p.key in selected_keys]

    if args.stage == "report":
        if not all_complete(state):
            raise RuntimeError("cannot report before every predeclared final cell is complete")
        build_report(state)
        return 0

    device = gpu_guard.acquire(min_free_gb=20)
    assert_gpu_zero(device)
    event(state, "run_started", stage=args.stage, pairs=selected_keys,
          gpu=torch.cuda.get_device_name(0), smoke=args.smoke)

    model_holders: dict[str, list] = {}
    for spec in specs:
        print(f"[{NAME}] pair={spec.key} grid={list(spec.layers)} shared={spec.shared_layer}",
              flush=True)
        frames = load_pair_frames(spec)
        state.setdefault("data", {})[spec.key] = {
            s: {"rows": len(df), "sha256_ordered_triples": frame_hash(df)}
            for s, df in frames.items()
        }
        save_state(state)
        holder = model_holders.setdefault(spec.model, [None, None])
        if args.smoke:
            selection_for_pair(spec, frames, state, holder, device, args, smoke=True)
            continue
        if args.stage in ("selection", "all") and spec.key not in state["selected_layers"]:
            selection_for_pair(spec, frames, state, holder, device, args)
        if args.stage in ("final", "all"):
            final_for_pair(spec, frames, state, holder, device, args)
        check_gpu_contention()
    model_holders.clear()
    torch.cuda.empty_cache()
    if not args.smoke and all_complete(state):
        build_report(state)
    event(state, "run_finished", stage=args.stage, pairs=selected_keys, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
