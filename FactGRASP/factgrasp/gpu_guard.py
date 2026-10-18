# picks a free gpu on a shared server and steps aside if another user's job shows up
# by default only your own processes count as yours

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from typing import Iterable, Optional

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

DEFAULT_ALLOWLIST = (getpass.getuser(),)
_extra = tuple(u.strip() for u in os.environ.get("FACTGRASP_SHARE_WITH", "").split(",") if u.strip())
if _extra:
    DEFAULT_ALLOWLIST = DEFAULT_ALLOWLIST + _extra
FREE_MB = 2048
EXIT_NO_GPU = 3
EXIT_YIELD = 4

_STATE = {"gpu": None, "allowlist": DEFAULT_ALLOWLIST, "min_free_gb": 20}


def _run(cmd: list) -> Optional[str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def _rows(text: Optional[str]) -> list:
    if not text:
        return []
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            out.append([c.strip() for c in line.split(",")])
    return out


def gpu_memory() -> list:
    txt = _run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits"])
    out = []
    for r in _rows(txt):
        try:
            out.append((int(r[0]), float(r[1]), float(r[2])))
        except (ValueError, IndexError):
            continue
    return out


def _uuid_to_index() -> dict:
    txt = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    m = {}
    for r in _rows(txt):
        try:
            m[r[1]] = int(r[0])
        except (ValueError, IndexError):
            continue
    return m


def compute_apps() -> list:
    txt = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits"])
    if txt is not None:
        idx = _uuid_to_index()
        out = []
        for r in _rows(txt):
            if len(r) < 3:
                continue
            try:
                out.append((idx.get(r[0]), int(r[1]), float(r[2])))
            except ValueError:
                continue
        return out
    txt = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"])
    out = []
    for r in _rows(txt):
        if len(r) < 2:
            continue
        try:
            out.append((None, int(r[0]), float(r[1])))
        except ValueError:
            continue
    return out


def pid_users(pids: Iterable[int]) -> dict:
    pids = sorted({int(p) for p in pids})
    if not pids:
        return {}
    txt = _run(["ps", "-o", "pid=,user:32=", "-p", ",".join(str(p) for p in pids)])
    users = {p: "?" for p in pids}
    if not txt:
        return users
    for line in txt.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                users[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return users


def survey(allowlist=DEFAULT_ALLOWLIST, min_free_gb: float = 20,
           share_allowlisted: bool = False) -> list:
    mem = gpu_memory()
    apps = compute_apps()
    users = pid_users([p for _, p, _ in apps])
    allow = {u.lower() for u in allowlist}
    recs = []
    for idx, used_mb, total_mb in mem:
        free_mb = total_mb - used_mb
        on_gpu = [(p, users.get(p, "?")) for gi, p, _ in apps if gi is None or gi == idx]
        owners = sorted({u for _, u in on_gpu})
        strangers = [u for u in owners if u.lower() not in allow]
        if used_mb < FREE_MB and not on_gpu:
            free, why = True, "idle"
        elif strangers:
            free, why = False, "foreign process: " + ",".join(strangers)
        elif not share_allowlisted:
            free, why = False, ("busy: %d process(es) owned by %s"
                                % (len(on_gpu), ",".join(owners) or "?"))
        elif free_mb >= min_free_gb * 1024:
            free, why = True, "allowlisted users only, enough headroom"
        else:
            free, why = False, "only %.1f GB free" % (free_mb / 1024)
        recs.append({"index": idx, "used_gb": used_mb / 1024, "total_gb": total_mb / 1024,
                     "free_gb": free_mb / 1024, "owners": owners, "strangers": strangers,
                     "free": free, "reason": why, "n_proc": len(on_gpu)})
    return recs


def report(allowlist=DEFAULT_ALLOWLIST, min_free_gb: float = 20,
           share_allowlisted: bool = False) -> list:
    recs = survey(allowlist, min_free_gb, share_allowlisted)
    if not recs:
        print("[gpu_guard] nvidia-smi returned nothing (no GPU, or driver not visible)")
    for r in recs:
        print("[gpu_guard] gpu %d: %.1f/%.1f GB used, %.1f GB free, procs=%d owners=%s -> %s (%s)"
              % (r["index"], r["used_gb"], r["total_gb"], r["free_gb"], r["n_proc"],
                 ",".join(r["owners"]) or "-", "FREE" if r["free"] else "busy", r["reason"]))
    return recs


def _pin(idx: int) -> str:
    uuid = {v: k for k, v in _uuid_to_index().items()}.get(idx)
    value = uuid if uuid else str(idx)
    pre = os.environ.get("CUDA_VISIBLE_DEVICES")
    already_masked = pre is not None and pre.strip() in (value, str(idx))
    if pre and not already_masked:
        raise RuntimeError(
            "gpu_guard: CUDA_VISIBLE_DEVICES is already %r in the environment but GPU %d is "
            "%r. Refusing rather than running on a card nobody asked for." % (pre, idx, value))
    tv = sys.modules.get("torch")
    if tv is not None and tv.cuda.is_initialized() and not already_masked:
        raise RuntimeError(
            "gpu_guard: CUDA was already initialised in this process before the pin, so "
            "CUDA_VISIBLE_DEVICES cannot take effect and the process would run on whichever "
            "card the runtime picked. Call acquire() before anything touches torch.cuda, or "
            "export CUDA_VISIBLE_DEVICES in the shell before python starts (both queue "
            "runners do this).")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


def _verify_name(idx: int) -> None:
    import torch
    if not torch.cuda.is_available():
        return
    txt = _run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"])
    smi = {int(r[0]): r[1] for r in _rows(txt) if len(r) >= 2 and r[0].isdigit()}
    seen = torch.cuda.get_device_name(0)
    want = smi.get(idx)
    if want and want.strip() != seen.strip() and torch.cuda.device_count() > 1:
        print("[gpu_guard] note: %d cards visible to torch, so CUDA_VISIBLE_DEVICES did not "
              "take effect in this process" % torch.cuda.device_count())
    if want and want.strip() != seen.strip():
        print("[gpu_guard] REFUSING: torch sees %r but nvidia-smi index %d is %r"
              % (seen, idx, want))
        sys.exit(EXIT_NO_GPU)
    print("[gpu_guard] torch device 0 = %s (nvidia-smi index %d)" % (seen, idx))


def _pick(free: list) -> dict:
    return sorted(free, key=lambda r: (r["reason"] != "idle", -r["free_gb"], r["index"]))[0]


def acquire(allowlist=DEFAULT_ALLOWLIST, min_free_gb: float = 20,
            allow_cpu: bool = False, dry: bool = False, verbose: bool = True,
            share_allowlisted: bool = False):
    forced = os.environ.get("FACTGRASP_FORCE_GPU")
    if forced is not None and forced != "":
        idx = int(forced)
        _STATE.update({"gpu": idx, "allowlist": tuple(allowlist), "min_free_gb": min_free_gb})
        if dry:
            print("[gpu_guard] dry run: FACTGRASP_FORCE_GPU=%d would be used" % idx)
            return None
        value = _pin(idx)
        import torch
        print("[gpu_guard] using GPU %d (forced by FACTGRASP_FORCE_GPU; CUDA_VISIBLE_DEVICES=%s)"
              % (idx, value))
        _verify_name(idx)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    recs = (report(allowlist, min_free_gb, share_allowlisted) if verbose
            else survey(allowlist, min_free_gb, share_allowlisted))
    free = [r for r in recs if r["free"]]
    _STATE.update({"allowlist": tuple(allowlist), "min_free_gb": min_free_gb})

    if dry:
        if free:
            pick = _pick(free)
            print("[gpu_guard] dry run: would take GPU %d (%.1f GB free, %s)"
                  % (pick["index"], pick["free_gb"], pick["reason"]))
        else:
            print("[gpu_guard] dry run: no free GPU right now")
        return None

    if not free:
        print("[gpu_guard] no free GPU (need %.0f GB and no foreign process)" % min_free_gb)
        if not allow_cpu:
            sys.exit(EXIT_NO_GPU)
        print("[gpu_guard] --allow-cpu given, falling back to CPU")
        _STATE["gpu"] = None
        import torch
        return torch.device("cpu")

    pick = _pick(free)
    _STATE["gpu"] = pick["index"]
    value = _pin(pick["index"])
    import torch
    if not torch.cuda.is_available():
        print("[gpu_guard] torch reports no CUDA device after masking")
        if not allow_cpu:
            sys.exit(EXIT_NO_GPU)
        return torch.device("cpu")
    print("[gpu_guard] using GPU %d (%.1f GB free; CUDA_VISIBLE_DEVICES=%s)"
          % (pick["index"], pick["free_gb"], value))
    _verify_name(pick["index"])
    return torch.device("cuda:0")


def check_yield() -> bool:
    idx = _STATE.get("gpu")
    if idx is None:
        return False
    for r in survey(_STATE["allowlist"], _STATE["min_free_gb"],
                    share_allowlisted=True):
        if r["index"] == idx and r["strangers"]:
            print("[gpu_guard] yielding GPU %d to %s" % (idx, ",".join(r["strangers"])))
            return True
    return False


def yield_and_exit(note: str = "") -> None:
    print("[gpu_guard] checkpoint saved, exiting with code %d. %s" % (EXIT_YIELD, note))
    sys.exit(EXIT_YIELD)


def current_gpu():
    return _STATE.get("gpu")


if __name__ == "__main__":
    acquire(dry=True)
