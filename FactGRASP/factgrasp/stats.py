# t intervals bootstrap and rank correlations used by the analysis scripts

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

try:
    from scipy import stats as _sps
except Exception:
    _sps = None

_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def _t_quantile(p: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if _sps is not None:
        return float(_sps.t.ppf(p, df))
    if abs(p - 0.975) < 1e-9:
        return _T95[df - 1] if df <= 30 else 1.96 + 2.4 / df
    z = _norm_quantile(p)
    return z * (1.0 + (z * z + 1.0) / (4.0 * df))


def _norm_quantile(p: float) -> float:
    if _sps is not None:
        return float(_sps.norm.ppf(p))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _t_sf(t: float, df: int) -> float:
    if _sps is not None:
        return float(_sps.t.sf(t, df))
    z = t / (1.0 + (t * t + 1.0) / (4.0 * df)) if df > 0 else t
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _mean_sd(values: Sequence[float]):
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(v) / n
    if n == 1:
        return mu, 0.0, 1
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, sd, n


def t_ci(values: Sequence[float], alpha: float = 0.05) -> dict:
    mu, sd, n = _mean_sd(values)
    if n <= 1:
        return {"mean": mu, "sd": 0.0, "n": n, "half_width": 0.0, "lo": mu, "hi": mu,
                "se": 0.0}
    se = sd / math.sqrt(n)
    hw = _t_quantile(1 - alpha / 2, n - 1) * se
    return {"mean": mu, "sd": sd, "n": n, "se": se, "half_width": hw,
            "lo": mu - hw, "hi": mu + hw}


def tost(diff_samples: Sequence[float], margin: float, alpha: float = 0.05) -> dict:
    mu, sd, n = _mean_sd(diff_samples)
    if n <= 1 or margin <= 0:
        return {"mean": mu, "n": n, "margin": margin, "p_lower": float("nan"),
                "p_upper": float("nan"), "p": float("nan"), "equivalent": False,
                "lo": mu, "hi": mu}
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    if se == 0.0:
        eq = abs(mu) < margin
        return {"mean": mu, "n": n, "margin": margin, "p_lower": 0.0 if eq else 1.0,
                "p_upper": 0.0 if eq else 1.0, "p": 0.0 if eq else 1.0,
                "equivalent": eq, "lo": mu, "hi": mu, "se": 0.0}
    df = n - 1
    t_low = (mu + margin) / se
    t_up = (margin - mu) / se
    p_low = _t_sf(t_low, df)
    p_up = _t_sf(t_up, df)
    q = _t_quantile(1 - alpha, df)
    return {"mean": mu, "n": n, "sd": sd, "se": se, "margin": margin,
            "p_lower": p_low, "p_upper": p_up, "p": max(p_low, p_up),
            "equivalent": bool(max(p_low, p_up) < alpha),
            "lo": mu - q * se, "hi": mu + q * se}


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list:
    p = [float(x) for x in pvalues]
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> list:
    return [a < alpha for a in holm(pvalues, alpha)]


def bootstrap_ci(values: Sequence[float], stat: Callable = None, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    v = [float(x) for x in values]
    n = len(v)
    if stat is None:
        def stat(xs):
            return sum(xs) / len(xs)
    if n == 0:
        return {"stat": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        draws.append(stat([v[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[max(0, int(math.floor(alpha / 2 * n_boot)))]
    hi = draws[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return {"stat": stat(v), "lo": lo, "hi": hi, "n": n, "n_boot": n_boot,
            "significant": not (lo <= 0.0 <= hi)}


def _rankdata(x: Sequence[float]) -> list:
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return float("nan")
    rx, ry = _rankdata(list(x)), _rankdata(list(y))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 2000,
                alpha: float = 0.05, seed: int = 0) -> dict:
    n = len(x)
    rho = spearman(x, y)
    if n < 3:
        return {"rho": rho, "lo": float("nan"), "hi": float("nan"), "n": n}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(r):
            draws.append(r)
    if not draws:
        return {"rho": rho, "lo": float("nan"), "hi": float("nan"), "n": n}
    draws.sort()
    b = len(draws)
    lo = draws[max(0, int(math.floor(alpha / 2 * b)))]
    hi = draws[min(b - 1, int(math.ceil((1 - alpha / 2) * b)) - 1)]
    return {"rho": rho, "lo": lo, "hi": hi, "n": n, "n_boot": b,
            "significant": not (lo <= 0.0 <= hi)}


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    conc = disc = 0
    tx = ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    den = ((n0 - tx) * (n0 - ty)) ** 0.5
    return float((conc - disc) / den) if den > 0 else 0.0
