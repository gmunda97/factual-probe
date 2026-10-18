# loads the hf models, embeds strings at chosen layers with an on disk cache, and the pca projector

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from factgrasp import paths

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

OUT_PROJ_NAMES = ("o_proj", "out_proj", "dense")
ATTN_NAMES = ("self_attn", "attention", "attn", "self_attention")


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace("@", "__at__")


def split_revision(model_id: str) -> tuple:
    if "@" in model_id:
        repo, rev = model_id.split("@", 1)
        return repo, rev
    return model_id, None


def hidden_index(module_idx: int) -> int:
    return module_idx + 1


def module_index(hidden_idx: int) -> int:
    return hidden_idx - 1


@dataclass
class EncoderSpec:
    model_id: str
    pooling: str = "last"
    dtype: str = "bf16"
    max_len: int = 32
    batch: int = 256

    def __post_init__(self) -> None:
        if self.pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {self.pooling!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {self.dtype!r}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @property
    def slug(self) -> str:
        return model_slug(self.model_id)


def text_config(config):
    if isinstance(config, dict):
        tc = config.get("text_config")
        return tc if isinstance(tc, dict) else config
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config


_MISSING = object()


def _cfg_attr(obj, key: str):
    try:
        if hasattr(obj, key):
            return getattr(obj, key)
        return _MISSING
    except AttributeError:
        return _MISSING
    except Exception:
        per = None
        try:
            per = getattr(obj, "per_layer_config", None)
        except Exception:
            per = None
        if per:
            vals = []
            for layer_cfg in per:
                v = getattr(layer_cfg, key, None)
                if v is not None:
                    vals.append(v)
            uniq = list(dict.fromkeys(vals))
            if len(uniq) == 1:
                return uniq[0]
            if uniq:
                return uniq
        return _MISSING


def cfg_get(config, key: str, default=None):
    tc = text_config(config)
    if isinstance(tc, dict):
        if key in tc:
            return tc[key]
    else:
        v = _cfg_attr(tc, key)
        if v is not _MISSING and v is not None:
            return v
    if isinstance(config, dict):
        return config.get(key, default)
    v = _cfg_attr(config, key)
    return default if v is _MISSING else v


def full_attention_layers(config) -> list[int]:
    layer_types = cfg_get(config, "layer_types", None)
    if layer_types:
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]
    n = cfg_get(config, "num_hidden_layers", None)
    if n is None:
        raise ValueError("config has neither layer_types nor num_hidden_layers")
    return list(range(int(n)))


def encoder_info(model_id: str) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    n_layers = int(cfg_get(cfg, "num_hidden_layers"))
    hidden = int(cfg_get(cfg, "hidden_size"))
    def _modal(v):
        if isinstance(v, (list, tuple)) and v:
            return max(set(v), key=list(v).count), True
        return v, False

    varies = []
    heads, v = _modal(cfg_get(cfg, "num_attention_heads", None))
    varies += ["num_attention_heads"] if v else []
    heads = int(heads) if heads is not None else None
    kv_heads, v = _modal(cfg_get(cfg, "num_key_value_heads", heads))
    varies += ["num_key_value_heads"] if v else []
    kv_heads = int(kv_heads) if kv_heads is not None else None
    head_dim, v = _modal(cfg_get(cfg, "head_dim", None))
    varies += ["head_dim"] if v else []
    if head_dim is None and heads:
        head_dim = hidden // heads
    layer_types = list(cfg_get(cfg, "layer_types", []) or [])
    arch = list(getattr(cfg, "architectures", []) or [])
    tc = text_config(cfg)
    return {
        "model_id": model_id,
        "model_type": getattr(cfg, "model_type", None),
        "text_model_type": (tc.get("model_type") if isinstance(tc, dict)
                            else getattr(tc, "model_type", None)),
        "architectures": arch,
        "multimodal_wrapper": any(a.endswith("ForConditionalGeneration") for a in arch),
        "n_layers": n_layers,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": int(head_dim) if head_dim is not None else None,
        "per_layer_fields_that_vary": varies,
        "layer_types": layer_types,
        "full_attention_layers": full_attention_layers(cfg),
        "num_kv_shared_layers": cfg_get(cfg, "num_kv_shared_layers", 0),
        "layer_grid": layer_grid(n_layers),
    }


def _layer_list(module) -> Optional[nn.ModuleList]:
    for name in ("layers", "h"):
        blocks = getattr(module, name, None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
            return blocks
    enc = getattr(module, "encoder", None)
    if enc is not None:
        for name in ("layer", "layers"):
            blocks = getattr(enc, name, None)
            if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
                return blocks
    return None


def text_model(model) -> nn.Module:
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None:
            candidates.append(lm)
    lm = getattr(model, "language_model", None)
    if lm is not None:
        candidates.append(lm)
    if inner is not None:
        candidates.append(inner)
    candidates.append(model)

    for cand in candidates:
        if _layer_list(cand) is not None:
            return cand

    want = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            want = int(cfg_get(cfg, "num_hidden_layers"))
        except Exception:
            want = None
    best = None
    for name, mod in model.named_modules():
        blocks = _layer_list(mod)
        if blocks is None:
            continue
        if want is not None and len(blocks) == want:
            return mod
        if best is None or len(blocks) > len(_layer_list(best)):
            best = mod
    if best is not None:
        return best
    raise AttributeError(
        "could not locate the text tower: no .layers, .h or .encoder.layer found under "
        f"{type(model).__name__}"
    )


def decoder_layers(model) -> nn.ModuleList:
    blocks = _layer_list(text_model(model))
    if blocks is None:
        raise AttributeError("text tower has no .layers, .h or .encoder.layer")
    return blocks


def load_decoder(model_id: str, device, dtype: str = "bf16",
                 prefer: str = "causal") -> tuple:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    quant = dtype if dtype in QUANT_MODES else ""
    base_dtype = "bf16" if quant else dtype

    repo, revision = split_revision(model_id)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False, **({"revision": revision} if revision else {}))
    ensure_pad_token(tok)

    order = [AutoModelForCausalLM, AutoModel]
    if prefer == "auto":
        order = order[::-1]

    errors = []
    model = None
    for cls in order:
        try:
            model = _from_pretrained(cls, model_id, DTYPES[base_dtype], quant)
            decoder_layers(model)
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
            model = None
    if model is None:
        raise RuntimeError(
            f"could not load {model_id}; attempts:\n  " + "\n  ".join(errors))

    model = (model if quant else model.to(device)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


QUANT_MODES = ("int8", "nf4")


def _quant_config(mode: str):
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes
    except ImportError as exc:
        raise RuntimeError(
            f"loading in {mode} needs bitsandbytes in this environment "
            f"(pip install bitsandbytes); {type(exc).__name__}: {exc}") from exc
    from transformers import BitsAndBytesConfig
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def _from_pretrained(cls, model_id: str, dtype: torch.dtype, quant: str = ""):
    repo, revision = split_revision(model_id)
    kw = {"trust_remote_code": False}
    if revision:
        kw["revision"] = revision
    if quant:
        kw["quantization_config"] = _quant_config(quant)
        kw["device_map"] = {"": 0}
    try:
        return cls.from_pretrained(repo, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(repo, torch_dtype=dtype, **kw)


def ensure_pad_token(tok) -> None:
    if getattr(tok, "pad_token", None) is not None:
        return
    for attr in ("eos_token", "unk_token", "bos_token"):
        val = getattr(tok, attr, None)
        if val is not None:
            tok.pad_token = val
            return
    tok.add_special_tokens({"pad_token": "<pad>"})


def layer_grid(n_layers: int) -> list[int]:
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    step = 2 if n_layers <= 36 else 3
    grid = list(range(step, n_layers + 1, step))
    if n_layers not in grid:
        grid.append(n_layers)
    if not grid:
        grid = [n_layers]
    return sorted(set(grid))


def refine(best: int, n_layers: Optional[int] = None) -> list[int]:
    out = []
    for l in (best - 1, best + 1):
        if l < 1:
            continue
        if n_layers is not None and l > n_layers:
            continue
        out.append(l)
    return out


@torch.no_grad()
# intermediate decoder states go through the model's own final norm before use (sec 3.3)
def final_norm(model) -> Optional[nn.Module]:
    tm = text_model(model)
    for name in ("norm", "final_layernorm", "ln_f", "final_layer_norm", "layer_norm", "norm_f"):
        mod = getattr(tm, name, None)
        if isinstance(mod, nn.Module):
            return mod
    return None


def embed_strings(texts: Sequence[str], tok, model, layers: Iterable[int],
                  pooling: str, batch: int = 256, max_len: int = 32,
                  device=None, progress: bool = True, frame: str = "final") -> dict:
    if pooling not in ("last", "mean"):
        raise ValueError(f"unknown pooling {pooling!r}")
    if frame not in ("raw", "final"):
        raise ValueError(f"unknown frame {frame!r}")
    layers = sorted(set(int(l) for l in layers))
    tm = text_model(model)
    fnorm = final_norm(model) if frame == "final" else None
    if device is None:
        try:
            device = next(tm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    ensure_pad_token(tok)

    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left" if pooling == "last" else "right"
    chunks = {l: [] for l in layers}
    iterator = range(0, len(texts), batch)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"embed[{pooling}]", leave=False)
        except ImportError:
            pass
    try:
        for i in iterator:
            part = list(texts[i:i + batch])
            enc = tok(part, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            out = tm(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError(
                    "the text tower returned no hidden_states; pass "
                    "output_hidden_states=True through to it")
            mask = enc["attention_mask"]
            for l in layers:
                if l >= len(hs):
                    raise IndexError(
                        f"layer {l} out of range: only {len(hs)} hidden states")
                h = hs[l]
                if fnorm is not None and l != len(hs) - 1:
                    h = fnorm(h)
                h = h.float()
                # decoders take the last non padding token (left padding so it is position -1)
                # encoders take a mask aware mean, the first token is dropped from the mean when a string has more than one token
                if pooling == "last":
                    pooled = h[:, -1, :]
                else:
                    m = mask.unsqueeze(-1).float()
                    if m.shape[1] > 1:
                        multi = (mask.sum(1) > 1).view(-1, 1, 1).float()
                        m = m.clone()
                        m[:, :1, :] = m[:, :1, :] * (1.0 - multi)
                    pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
                chunks[l].append(pooled.to(torch.float16).cpu())
    finally:
        tok.padding_side = old_side
    return {l: torch.cat(chunks[l], dim=0) for l in layers}


def cache_path(model_id: str, dataset: str, split: str, field: str,
               pooling: str, layer: int, frame: str = "final") -> str:
    d = os.path.join(paths.CACHE_DEC, model_slug(model_id))
    os.makedirs(d, exist_ok=True)
    suffix = "" if frame == "final" else f"_{frame}"
    return os.path.join(d, f"{dataset}_{split}_{field}_{pooling}{suffix}_L{layer}.pt")


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_or_embed(texts: Sequence[str], tok, model, layers: Iterable[int],
                  pooling: str, model_id: str, dataset: str, split: str,
                  field: str, batch: int = 256, max_len: int = 32, device=None,
                  force: bool = False, progress: bool = True,
                  frame: str = "final") -> dict:
    layers = sorted(set(int(l) for l in layers))
    have, missing = {}, []
    for l in layers:
        p = cache_path(model_id, dataset, split, field, pooling, l, frame)
        if os.path.exists(p) and not force:
            try:
                t = torch.load(p, map_location="cpu")
            except Exception:
                t = None
            if t is not None and t.shape[0] == len(texts):
                have[l] = t
                continue
        missing.append(l)
    if missing:
        fresh = embed_strings(texts, tok, model, missing, pooling, batch=batch,
                              max_len=max_len, device=device, progress=progress,
                              frame=frame)
        for l, v in fresh.items():
            _atomic_save(v, cache_path(model_id, dataset, split, field, pooling, l, frame))
            have[l] = v
    return {l: have[l] for l in layers}


class PCAProjector:

    def __init__(self, mean: torch.Tensor, components: torch.Tensor) -> None:
        self.mean = mean.float()
        self.components = components.float()

    @property
    def k(self) -> int:
        return int(self.components.shape[1])

    @property
    def d_in(self) -> int:
        return int(self.components.shape[0])

    @classmethod
    # pca to 768 dims fitted on training rows only, for models wider than 768
    def fit(cls, X: torch.Tensor, k: int = 768, sample: int = 20000,
            seed: int = 0, niter: int = 4) -> "PCAProjector":
        Xf = X.float()
        if Xf.ndim != 2:
            raise ValueError("PCAProjector.fit expects a (N, D) matrix")
        n, d = Xf.shape
        if n > sample:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n, generator=g)[:sample]
            S = Xf[idx]
        else:
            S = Xf
        mean = S.mean(dim=0)
        q = int(min(k, d, S.shape[0]))
        _, _, V = torch.pca_lowrank(S - mean, q=q, center=False, niter=niter)
        return cls(mean, V[:, :q].contiguous())

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X.float() - self.mean) @ self.components

    __call__ = transform

    def state_dict(self) -> dict:
        return {"mean": self.mean, "components": self.components}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "PCAProjector":
        return cls(sd["mean"], sd["components"])

    def save(self, path: str) -> None:
        _atomic_save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "PCAProjector":
        return cls.from_state_dict(torch.load(path, map_location="cpu"))


def attention_module(layer: nn.Module) -> nn.Module:
    for name in ATTN_NAMES:
        mod = getattr(layer, name, None)
        if mod is not None and isinstance(mod, nn.Module):
            if any(getattr(mod, p, None) is not None for p in OUT_PROJ_NAMES):
                return mod
    for name in ATTN_NAMES:
        mod = getattr(layer, name, None)
        if isinstance(mod, nn.Module):
            return mod
    return layer


def out_projection(layer: nn.Module) -> nn.Module:
    attn = attention_module(layer)
    for name in OUT_PROJ_NAMES:
        mod = getattr(attn, name, None)
        if isinstance(mod, nn.Module):
            return mod
    for name, mod in attn.named_modules():
        if name.split(".")[-1] in OUT_PROJ_NAMES:
            return mod
    raise AttributeError(
        f"no attention output projection ({' | '.join(OUT_PROJ_NAMES)}) under "
        f"{type(attn).__name__}")


class HeadContextHook:

    def __init__(self, model, layer_idx: int, pooling: str = "last",
                 num_heads: Optional[int] = None) -> None:
        if pooling not in ("last", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.layer_idx = int(layer_idx)
        layers = _layer_list(model)
        if layers is None:
            layers = decoder_layers(model)
        self.layer = layers[self.layer_idx]
        self.proj = out_projection(self.layer)
        if num_heads is None:
            cfg = getattr(model, "config", None)
            num_heads = cfg_get(cfg, "num_attention_heads", None) if cfg is not None else None
        if num_heads is None:
            raise ValueError("num_heads could not be read from the config; pass it")
        self.num_heads = int(num_heads)
        self._buf = None
        self._mask = None
        self._handle = self.proj.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = args[0] if isinstance(args, tuple) else args
        if not torch.is_tensor(x):
            return None
        h = x.float()
        if h.ndim == 2:
            pooled = h
        elif self.pooling == "last" or self._mask is None:
            pooled = h[:, -1, :]
        else:
            m = self._mask.to(h.device).unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        B, W = pooled.shape
        if W % self.num_heads != 0:
            raise RuntimeError(
                f"o_proj input width {W} is not divisible by {self.num_heads} heads")
        self._buf = pooled.view(B, self.num_heads, W // self.num_heads).detach().cpu()
        return None

    def set_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._mask = attention_mask

    def pop(self) -> Optional[torch.Tensor]:
        b, self._buf = self._buf, None
        return b

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "HeadContextHook":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
