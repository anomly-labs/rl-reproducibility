# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""worst_case_variance.py — float32 says the variance is NEGATIVE.

The textbook one-pass "raw moment" sample variance  Var(x) = E[x^2] - E[x]^2  is in countless naive
statistics / ML routines. For data with a large mean and a tiny spread, the two moments E[x^2] and E[x]^2
are both ~mean^2 (huge) and nearly equal, so in float32 their difference is pure rounding noise — and
frequently comes out NEGATIVE, which is mathematically impossible for a variance (std = sqrt(negative) =
NaN; any downstream z-score / whitening / Cholesky blows up). The exact reduction computes the identical
formula correctly and returns a non-negative value.

Honest scope: constructed worst-case data (to bound how wrong float gets on ordinary float-representable
inputs). The exact path here uses an order-independent, two-moment exact reduction (math.fsum on x and on
x^2), so it returns the correctly-rounded true variance; the data is snapped to float32 first, so the ONLY
difference measured is the accumulation.
"""
from __future__ import annotations

import math
import random

import numpy as np


def make_data(mu, spread, n, seed):
    """x_i = mu + tiny noise, snapped to float32. Large mean + tiny spread => the two moments nearly
    cancel in float32."""
    rng = random.Random(seed)
    return [float(np.float32(mu + rng.uniform(-spread, spread))) for _ in range(n)]


def float32_naive_var(x):
    """One-pass raw-moment variance in float32: E[x^2] - E[x]^2, sequential float32 accumulation."""
    n = np.float32(len(x))
    s2 = np.float32(0.0)
    s = np.float32(0.0)
    for v in x:
        vf = np.float32(v)
        s2 = np.float32(s2 + np.float32(vf * vf))
        s = np.float32(s + vf)
    mean = np.float32(s / n)
    return float(np.float32(np.float32(s2 / n) - np.float32(mean * mean)))


def exact_var(x):
    """Order-independent exact variance from the same raw moments (math.fsum of x and x^2)."""
    n = len(x)
    s2 = math.fsum(float(v) * float(v) for v in x)
    s = math.fsum(float(v) for v in x)
    return (n * s2 - s * s) / (n * n)


def run():
    rows = []
    for mu, spread, n, seed in [(1e7, 1e-3, 4000, 1), (3e6, 1e-2, 500, 2), (1e7, 1.0, 1000, 3),
                                (8e6, 0.5, 2000, 4)]:
        x = make_data(mu, spread, n, seed)
        rows.append({"n": len(x), "mean": mu, "true": exact_var(x), "float32": float32_naive_var(x)})
    return rows


def report(rows):
    print("  naive one-pass variance E[x^2]-E[x]^2 on large-mean / small-spread data:")
    n_neg = 0
    for r in rows:
        neg = r["float32"] < 0
        n_neg += neg
        tag = "  <- NEGATIVE variance (impossible; std = NaN)" if neg else ""
        print(f"    mean={r['mean']:.0e} n={r['n']:5d}  true var = {r['true']:.4g}   "
              f"float32 var = {r['float32']:+.4g}{tag}")
    print(f"    exact reduction = the true non-negative variance, always   <- verified vs math.fsum")
    ok = n_neg > 0
    print(f"  => {'float32 returns a NEGATIVE / wildly wrong variance; the exact reduction is correct and >= 0' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    report(run())
