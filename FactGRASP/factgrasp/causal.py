# prompt templates generation and residual stream hooks, used by the prompting baselines (table 13) and the intervention cost columns (table 12)

from __future__ import annotations

import math

import re
import string
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

WIKIDATA_TEMPLATES: dict[str, str] = {
    "instance of": "{s} is an instance of",
    "cast member": "One of the cast members of {s} is",
    "shares border with": "{s} shares a border with",
    "member of sports team": "{s} plays for the team",
    "occupation": "The occupation of {s} is",
    "country": "{s} is located in the country of",
    "country of citizenship": "{s} is a citizen of",
    "located in the administrative territorial entity": "{s} is located in",
    "located in time zone": "{s} is located in the time zone",
    "place of birth": "{s} was born in",
    "genre": "The genre of {s} is",
    "award received": "{s} received the award",
    "given name": "The given name of {s} is",
    "contains the administrative territorial entity": "{s} contains the territory of",
    "place of death": "{s} died in",
    "educated at": "{s} was educated at",
    "languages spoken, written or signed": "The language spoken by {s} is",
    "sibling": "The sibling of {s} is",
    "sport": "{s} plays the sport of",
    "country of origin": "{s} comes from the country of",
    "twinned administrative body": "{s} is twinned with",
    "member of": "{s} is a member of",
    "original language of film or TV show": "The original language of {s} is",
    "participant in": "{s} participated in",
    "described by source": "{s} is described by the source",
    "taxon rank": "The taxon rank of {s} is",
    "director": "{s} was directed by",
    "composer": "The music of {s} was composed by",
    "child": "The child of {s} is",
    "producer": "{s} was produced by",
}

CONCEPTNET_TEMPLATES: dict[str, str] = {
    "antonym of": "The opposite of {s} is",
    "capable of": "{s} is capable of",
    "has a": "{s} has a",
    "at location": "You would find {s} at",
    "causes": "{s} causes",
    "causes desire": "{s} makes you want",
    "distinct from": "{s} is distinct from",
    "desires": "{s} desires",
    "created by": "{s} was created by",
}

DATASET_TEMPLATES: dict[str, dict[str, str]] = {
    "wikidata": WIKIDATA_TEMPLATES,
    "conceptnet": CONCEPTNET_TEMPLATES,
}

GENERIC_TEMPLATE = "{s}: {r} is"


def fill_template(subject: str, relation: str, dataset: str = "wikidata") -> str:
    table = DATASET_TEMPLATES.get(dataset, {})
    tpl = table.get(relation)
    if tpl is None:
        return GENERIC_TEMPLATE.format(s=subject, r=relation)
    return tpl.format(s=subject)


def build_prompts(subjects: Sequence[str], relations: Sequence[str],
                  dataset: str = "wikidata") -> list[str]:
    return [fill_template(s, r, dataset) for s, r in zip(subjects, relations)]


def template_coverage(relations: Sequence[str], dataset: str = "wikidata") -> float:
    table = DATASET_TEMPLATES.get(dataset, {})
    if not len(relations):
        return 0.0
    return sum(1 for r in relations if r in table) / float(len(relations))


_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_ARTICLES = {"a", "an", "the"}


def normalise_answer(text: str) -> str:
    if text is None:
        return ""
    t = text.lower().translate(_PUNCT_TABLE)
    toks = [w for w in re.split(r"\s+", t) if w and w not in _ARTICLES]
    return " ".join(toks)


def contains_gold(generation: str, gold: str) -> bool:
    g, o = normalise_answer(generation), normalise_answer(gold)
    return bool(o) and o in g


def _text_model(model):
    try:
        from factgrasp import encoders
        return encoders.text_model(model)
    except Exception:
        for attr in ("language_model", "model"):
            sub = getattr(model, attr, None)
            if sub is not None:
                if hasattr(sub, "layers"):
                    return sub
                inner = getattr(sub, "language_model", None)
                if inner is not None and hasattr(inner, "layers"):
                    return inner
        if hasattr(model, "layers"):
            return model
        raise AttributeError("could not locate the module owning .layers")


def load_model_and_tokenizer(model_id: str, device):
    from transformers import AutoTokenizer

    tok = None
    model = None
    try:
        from factgrasp import encoders
        out = encoders.load_decoder(model_id, device)
        if isinstance(out, tuple):
            a, b = out[0], out[1]
            if hasattr(a, "encode"):
                tok, model = a, b
            else:
                tok, model = b, a
        else:
            model = out
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, trust_remote_code=False).to(device).eval()
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def residual_hook(text_model, layer_idx: int, fn: Callable[[torch.Tensor], torch.Tensor]):
    layer = text_model.layers[layer_idx]

    def _hook(module, args, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + tuple(output[1:])
        return fn(output)

    return layer.register_forward_hook(_hook)


def rescale_to_norm(v: torch.Tensor, target_norm) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    if not torch.is_tensor(target_norm):
        target_norm = torch.as_tensor(float(target_norm), dtype=v.dtype, device=v.device)
    target_norm = target_norm.to(v.dtype).to(v.device)
    if target_norm.dim() and target_norm.shape[-1] != 1:
        target_norm = target_norm.unsqueeze(-1)
    scale = torch.where(n > 0, target_norm / n.clamp_min(1e-12), torch.ones_like(n))
    return v * scale


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_direction(dim: int, seed: int = 0, n: int = 1,
                     ref_norm: Optional[torch.Tensor] = None,
                     dtype=torch.float32, device="cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    v = torch.randn(n, dim, generator=g).to(dtype)
    if ref_norm is not None:
        v = rescale_to_norm(v, ref_norm.to(v.dtype) if torch.is_tensor(ref_norm) else ref_norm)
    return v.to(device)


def shuffled_directions(dirs: torch.Tensor, seed: int = 0) -> torch.Tensor:
    n = dirs.shape[0]
    if n < 2:
        return dirs.clone()
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    if any(perm[i] == i for i in range(n)):
        perm = [(i + 1) % n for i in range(n)]
    return dirs[torch.tensor(perm, dtype=torch.long)].clone()


def gold_token_ids(tok, objects: Sequence[str]) -> torch.Tensor:
    ids = []
    for o in objects:
        enc = tok(" " + str(o), add_special_tokens=False)["input_ids"]
        ids.append(int(enc[0]) if len(enc) else int(tok.unk_token_id or 0))
    return torch.tensor(ids, dtype=torch.long)


def pick_foils(relations: Sequence[str], objects: Sequence[str], seed: int = 0) -> list[str]:
    g = torch.Generator().manual_seed(int(seed))
    by_rel: dict[str, list[int]] = {}
    for i, r in enumerate(relations):
        by_rel.setdefault(r, []).append(i)
    out: list[str] = []
    n = len(objects)
    for i, r in enumerate(relations):
        cands = [j for j in by_rel[r] if objects[j] != objects[i]]
        if not cands:
            cands = [j for j in range(n) if objects[j] != objects[i]]
        if not cands:
            out.append(str(objects[i]))
            continue
        pick = int(torch.randint(len(cands), (1,), generator=g).item())
        out.append(str(objects[cands[pick]]))
    return out


def subject_end_offsets(tok, prompts: Sequence[str], subjects: Sequence[str]) -> torch.Tensor:
    offs = []
    for p, s in zip(prompts, subjects):
        pos = p.find(str(s))
        if pos < 0:
            offs.append(-1)
            continue
        prefix = p[: pos + len(str(s))]
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        n_full = len(tok(p, add_special_tokens=False)["input_ids"])
        off = n_prefix - n_full - 1
        offs.append(int(min(-1, off)))
    return torch.tensor(offs, dtype=torch.long)


@torch.no_grad()
def steer(model, tok, prompts: Sequence[str], direction: torch.Tensor,
          alpha_list: Sequence[float], layer_idx: int, position: str = "last",
          offsets: Optional[torch.Tensor] = None, batch_size: int = 16,
          reduce_fn: Optional[Callable[[torch.Tensor], object]] = None,
          device=None) -> dict:
    if position not in ("last", "subject_last"):
        raise ValueError(f"unknown position {position!r}")
    if position == "subject_last" and offsets is None:
        raise ValueError("position='subject_last' needs offsets from subject_end_offsets")
    dev = device if device is not None else next(model.parameters()).device
    tm = _text_model(model)
    n = len(prompts)
    if direction.dim() == 1:
        direction = direction.unsqueeze(0).expand(n, -1)
    if direction.shape[0] != n:
        raise ValueError(f"direction has {direction.shape[0]} rows for {n} prompts")
    if offsets is None:
        offsets = torch.full((n,), -1, dtype=torch.long)

    out: dict = {}
    for alpha in alpha_list:
        chunks = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            enc = tok(list(prompts[start:stop]), return_tensors="pt", padding=True,
                      truncation=True, max_length=64).to(dev)
            T = enc["input_ids"].shape[1]
            idx = (T + offsets[start:stop]).clamp(0, T - 1).to(dev)
            d = unit(direction[start:stop].float()).to(dev)
            rows = torch.arange(stop - start, device=dev)

            def _fn(h, idx=idx, d=d, rows=rows, alpha=float(alpha)):
                if alpha == 0.0:
                    return h
                cur = h[rows, idx].float()
                add = (alpha * cur.norm(dim=-1, keepdim=True)) * d
                h = h.clone()
                h[rows, idx] = (cur + add).to(h.dtype)
                return h

            handle = residual_hook(tm, layer_idx, _fn) if float(alpha) != 0.0 else None
            try:
                logits = model(**enc).logits[:, -1, :].float().cpu()
            finally:
                if handle is not None:
                    handle.remove()
            chunks.append(logits)
        full = torch.cat(chunks, dim=0)
        out[float(alpha)] = reduce_fn(full) if reduce_fn is not None else full
    return out


def steering_metrics(logits: torch.Tensor, base_logits: torch.Tensor,
                     gold_ids: torch.Tensor, foil_ids: torch.Tensor) -> dict:
    lg = logits.float()
    lb = base_logits.float()
    logp = F.log_softmax(lg, dim=-1)
    logp_b = F.log_softmax(lb, dim=-1)
    n = lg.shape[0]
    ar = torch.arange(n)
    lp_gold = logp[ar, gold_ids]
    p_gold = lp_gold.exp()
    gold_logit = lg[ar, gold_ids]
    foil_logit = lg[ar, foil_ids]
    rank = (lg > gold_logit.unsqueeze(1)).sum(1) + 1
    p = logp.exp()
    kl = (p * (logp - logp_b)).sum(-1)
    rows = {"p_gold": p_gold, "logp_gold": lp_gold, "rank_gold": rank.float(),
            "logit_diff": gold_logit - foil_logit, "kl": kl}
    out = {f"{k}_mean": float(v.mean()) for k, v in rows.items()}
    out["hits1"] = float((rank == 1).float().mean())
    out["hits10"] = float((rank <= 10).float().mean())
    out["mrr"] = float((1.0 / rank.float()).mean())
    out["rows"] = rows
    return out


IDENTITY_PROMPT = "cat -> cat; 1135 -> 1135; hello -> hello; ? ->"


@torch.no_grad()
def patchscope_decode(model, tok, vectors: torch.Tensor, layer_idx: int,
                      gold_ids: Optional[torch.Tensor] = None,
                      prompt: str = IDENTITY_PROMPT, batch_size: int = 16,
                      target_norm: Optional[float] = None, device=None) -> dict:
    dev = device if device is not None else next(model.parameters()).device
    tm = _text_model(model)

    enc1 = tok(prompt, return_tensors="pt")
    ids = enc1["input_ids"][0].tolist()
    patch_index = None
    for i, t in enumerate(ids):
        if "?" in tok.decode([t]):
            patch_index = i
    if patch_index is None:
        raise ValueError("the identity prompt has no '?' token to patch")

    if target_norm is None:
        grab: dict = {}

        def _grab(h):
            grab["n"] = h[:, patch_index].float().norm(dim=-1).mean().item()
            return h

        _norm_layer = layer_idx if isinstance(layer_idx, int) else list(layer_idx)[0]
        handle = residual_hook(tm, _norm_layer, _grab)
        try:
            model(**{k: v.to(dev) for k, v in enc1.items()})
        finally:
            handle.remove()
        target_norm = float(grab["n"])

    n = vectors.shape[0]
    argmax, maxp, pg, rk = [], [], [], []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        b = stop - start
        enc = {k: v.repeat(b, 1).to(dev) for k, v in enc1.items()}
        vec = rescale_to_norm(vectors[start:stop].float(), target_norm).to(dev)

        def _fn(h, vec=vec):
            h = h.clone()
            h[:, patch_index] = vec.to(h.dtype)
            return h

        handle = residual_hook(tm, layer_idx, _fn)
        try:
            logits = model(**enc).logits[:, -1, :].float().cpu()
        finally:
            handle.remove()
        probs = logits.softmax(-1)
        am = probs.argmax(-1)
        argmax.append(am)
        maxp.append(probs.max(-1).values)
        if gold_ids is not None:
            gi = gold_ids[start:stop]
            ar = torch.arange(b)
            pg.append(probs[ar, gi])
            rk.append((logits > logits[ar, gi].unsqueeze(1)).sum(1) + 1)

    out = {"argmax": torch.cat(argmax), "max_prob": torch.cat(maxp),
           "target_norm": float(target_norm), "patch_index": int(patch_index)}
    if gold_ids is not None:
        hit = (out["argmax"] == gold_ids).float()
        out["hit"] = hit
        out["decode_rate"] = float(hit.mean())
        out["p_gold"] = torch.cat(pg)
        out["rank_gold"] = torch.cat(rk).float()
    return out


@torch.no_grad()
def perplexity(model, tok, sentences: Sequence[str], layer_idx: Optional[int] = None,
               direction: Optional[torch.Tensor] = None, alpha: float = 0.0,
               batch_size: int = 8, max_len: int = 256, device=None,
               positions: str = "all") -> dict:
    dev = device if device is not None else next(model.parameters()).device
    tm = _text_model(model)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    total_nll, total_tok = 0.0, 0
    try:
        for i in range(0, len(sentences), batch_size):
            part = [s for s in sentences[i:i + batch_size] if s and s.strip()]
            if not part:
                continue
            enc = tok(part, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len)
            enc = {k: v.to(dev) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            handle = None
            if direction is not None and alpha:
                d = direction.to(dev).float()
                if d.dim() == 1:
                    d = d.unsqueeze(0)
                unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)

                def _add(h, unit=unit, alpha=alpha, positions=positions):
                    hf = h.float()
                    u = unit[0].view(1, 1, -1)
                    if positions == "all":
                        return (hf + alpha * hf.norm(dim=-1, keepdim=True) * u).to(h.dtype)
                    out = hf.clone()
                    n0 = hf[:, :1].norm(dim=-1, keepdim=True)
                    out[:, :1] = hf[:, :1] + alpha * n0 * u
                    return out.to(h.dtype)

                handle = residual_hook(tm, layer_idx, _add)
            try:
                out = model(**enc)
            finally:
                if handle is not None:
                    handle.remove()
            logits = out.logits[:, :-1].float()
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            lp = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                reduction="none").view(target.shape)
            total_nll += float(lp[mask].sum())
            total_tok += int(mask.sum())
    finally:
        tok.padding_side = old_side
    mean_nll = total_nll / max(total_tok, 1)
    return {"nll": mean_nll, "perplexity": float(math.exp(min(mean_nll, 50.0))),
            "n_tokens": total_tok, "n_sentences": len(sentences)}


@torch.no_grad()
def patchscope_contains(model, tok, vectors: torch.Tensor, layer_idx, 
                        golds: Sequence[str], prompt: str = IDENTITY_PROMPT,
                        max_new_tokens: int = 8, batch_size: int = 16,
                        target_norm: Optional[float] = None, device=None) -> dict:
    dev = device if device is not None else next(model.parameters()).device
    tm = _text_model(model)
    enc1 = tok(prompt, return_tensors="pt")
    ids = enc1["input_ids"][0].tolist()
    patch_index = None
    for i, t in enumerate(ids):
        if "?" in tok.decode([t]):
            patch_index = i
    if patch_index is None:
        raise ValueError("the identity prompt has no '?' token to patch")

    if target_norm is None:
        grab = {}

        def _grab(h):
            grab["n"] = h[:, patch_index].float().norm(dim=-1).mean().item()
            return h

        handle = residual_hook(tm, layer_idx if isinstance(layer_idx, int)
                                     else list(layer_idx)[0], _grab)
        try:
            model(**{k: v.to(dev) for k, v in enc1.items()})
        finally:
            handle.remove()
        target_norm = float(grab["n"])

    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    hits, gens = [], []
    try:
        for start in range(0, vectors.shape[0], batch_size):
            stop = min(start + batch_size, vectors.shape[0])
            b = stop - start
            enc = {k: v.repeat(b, 1).to(dev) for k, v in enc1.items()}
            vec = rescale_to_norm(vectors[start:stop].float(), target_norm).to(dev)
            n_in = enc["input_ids"].shape[1]

            def _fn(h, vec=vec, n_in=n_in):
                if h.shape[1] < n_in:
                    return h
                h = h.clone()
                h[:, patch_index] = vec.to(h.dtype)
                return h

            layers = [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)
            handles = [residual_hook(tm, li, _fn) for li in layers]
            try:
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id)
            finally:
                for hd in handles:
                    hd.remove()
            for j in range(b):
                text = tok.decode(out[j][n_in:], skip_special_tokens=True)
                gens.append(text)
                hits.append(1.0 if contains_gold(text, golds[start + j]) else 0.0)
    finally:
        tok.padding_side = old_side
    h = torch.tensor(hits)
    return {"contains_rate": float(h.mean()), "hit": h, "generations": gens,
            "target_norm": float(target_norm), "patch_index": int(patch_index),
            "patch_layers": [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)}


@torch.no_grad()
def subject_edit_flip(model, tok, prompts: Sequence[str],
                      subject_vecs: Optional[torch.Tensor],
                      layer_idx: int, offsets: torch.Tensor,
                      target_ids: torch.Tensor, gold_ids: torch.Tensor,
                      batch_size: int = 16, device=None) -> dict:
    dev = device if device is not None else next(model.parameters()).device
    tm = _text_model(model)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    flips, to_gold, argmaxes, logit_rows = [], [], [], []
    try:
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            part = list(prompts[start:stop])
            enc = tok(part, return_tensors="pt", padding=True, truncation=True,
                      max_length=64)
            enc = {k: v.to(dev) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            off = offsets[start:stop].to(dev)
            vec = (subject_vecs[start:stop].to(dev).float()
                   if subject_vecs is not None else None)
            mask = enc["attention_mask"]
            last_real = mask.sum(dim=1) - 1
            first_real = (mask.shape[1] - mask.sum(dim=1)) if tok.padding_side == "left" \
                else torch.zeros_like(last_real)
            if tok.padding_side == "left":
                last_real = torch.full_like(last_real, mask.shape[1] - 1)
            pos_idx = (last_real + 1 + off).clamp_min(first_real)
            pos_idx = torch.minimum(pos_idx, last_real)
            if int((mask.gather(1, pos_idx.unsqueeze(1)) == 0).sum()) > 0:
                raise ValueError("subject_edit_flip chose a padding position; the offsets "
                                 "and the padding side disagree")

            def _fn(h, pos_idx=pos_idx, vec=vec):
                h = h.clone()
                h[torch.arange(h.shape[0], device=h.device), pos_idx] = vec.to(h.dtype)
                return h

            handle = residual_hook(tm, layer_idx, _fn) if vec is not None else None
            try:
                logits = model(**enc).logits[:, -1, :].float().cpu()
            finally:
                if handle is not None:
                    handle.remove()
            am = logits.argmax(-1)
            argmaxes.append(am)
            logit_rows.append(logits)
            flips.append((am == target_ids[start:stop]).float())
            to_gold.append((am == gold_ids[start:stop]).float())
    finally:
        tok.padding_side = old_side
    f = torch.cat(flips)
    return {"flip_rate": float(f.mean()), "flip": f,
            "still_gold_rate": float(torch.cat(to_gold).mean()),
            "argmax": torch.cat(argmaxes),
            "logits": torch.cat(logit_rows) if logit_rows else None}
