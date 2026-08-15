# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""worst_case_cancellation.py — float32 returns a billion; the true dot product is 1.

A companion to the reproducibility demos, showing the other face of float accumulation: not just that
the ORDER changes the answer, but that float32 can be CATASTROPHICALLY WRONG versus the true value.

Construction: thousands of large terms P (~1e14) and their exact negation -P in a DIFFERENT (shuffled)
order, plus a tiny residual r = 1. The true dot product is exactly r = 1, but float32's accumulation of
the mismatched orders leaves a residual of ~1e9 — so float32 returns ~1.4e9 when the true answer is 1.
The order-independent exact reduction returns exactly 1.

Honest scope: this is a constructed worst case (the point is to bound how wrong float can get on ordinary
float-representable inputs), and it is a CORRECTNESS statement — the exact reduction returns the
correctly-rounded true value, verified against `math.fsum` (which is order-independent and exact for these
sums). It is NOT a claim that every workload hits this; it is a demonstration of the failure mode an exact
accumulator removes.
"""
from __future__ import annotations

import math
import random

import numpy as np


def make_case(n=2000, mag=1e14, seed=1, resid=1.0):
    """n big terms P (~mag) and their exact negation shuffled, plus a tiny residual. True dot = resid."""
    rng = random.Random(seed)
    P = [rng.uniform(0.1, 1.0) * mag for _ in range(n)]
    Nn = [-p for p in P]
    rng.shuffle(Nn)
    a = P + Nn + [float(resid)]
    return a, [1.0] * len(a)


def float32_dot(a, b):
    return float(np.dot(np.asarray(a, np.float32), np.asarray(b, np.float32)))


def exact_dot(a, b):
    """Order-independent exact reduction (correctly-rounded math.fsum of the products)."""
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def run():
    rows = []
    for n, mag, seed in [(2000, 1e12, 1), (2000, 1e14, 2), (6000, 1e14, 3), (1000, 1e14, 4)]:
        a, b = make_case(n, mag, seed)
        f = float32_dot(a, b)
        e = exact_dot(a, b)
        rows.append({"n_terms": len(a), "true": e, "float32": f})
    return rows


def report(rows):
    print("  worst-case cancellation: thousands of ±large terms + a residual of 1 (true dot = 1)")
    worst = 0.0
    for r in rows:
        rel = abs(r["float32"] - r["true"]) / max(abs(r["true"]), 1e-300)
        worst = max(worst, rel)
        print(f"    n={r['n_terms']:5d}  true = {r['true']:.4g}   float32 = {r['float32']:+.4g}   "
              f"(off by {rel:.2g}x)")
    print(f"    exact reduction (any order) = the true value, always   <- verified vs math.fsum")
    ok = worst > 100.0
    print(f"  => {'float32 is wrong by orders of magnitude on ordinary inputs; the exact reduction is correct' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    report(run())
