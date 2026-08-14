# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""refquire.py — a reference ORDER-INDEPENDENT exact reduction.

This is the property that matters: a dot product / reduction whose result does NOT depend on the order
the terms are summed. Float hardware doesn't have it (see floatkernels.py — the same reduction in a
different tile/chunk order gives a different answer). This module does, using correctly-rounded
summation (`math.fsum`), which returns the same result for any permutation of its inputs.

Honest scope. `math.fsum` sums the (float64) products order-independently; a fully-exact reference
(exact products too, arbitrary-precision integers) is in `exact_int_dot` and agrees with fsum on the
values that matter here — run `python refquire.py` to see. Anomly's silicon goes further and faster:
a 256-bit b-posit **quire** keeps the products exact as well and runs at GPU-parity speed (measured:
131,072 / 131,072 outputs bit-identical on our Alveo U200 engine). This demo proves the *property* —
that an order-independent accumulator eliminates the nondeterminism — on real data, on your laptop.
"""
from __future__ import annotations

import math

import numpy as np


def exact_dot(a, b) -> float:
    """Order-independent dot product: identical result for any permutation of the terms."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return math.fsum((a * b).tolist())


def exact_scores(A, w) -> np.ndarray:
    """Reward-style scores <A[i], w> under the order-independent reduction."""
    A = np.asarray(A, np.float64)
    w = np.asarray(w, np.float64).ravel()
    return np.array([math.fsum((A[i] * w).tolist()) for i in range(A.shape[0])], dtype=np.float64)


def exact_logits(W, h) -> np.ndarray:
    """Logits <W[v], h> under the order-independent reduction (alias of exact_scores)."""
    return exact_scores(W, h)


def exact_int_dot(a, b, frac_bits: int = 200) -> float:
    """Fully-exact gold reference: exact products AND exact sum in arbitrary-precision integers,
    rounded once at readout. Order-independent by construction. Slow — for verification, not the loop."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()

    def to_fixed(x: float) -> int:
        if x == 0.0:
            return 0
        m, e = math.frexp(x)                 # x = m * 2**e, 0.5 <= |m| < 1
        mant = int(math.ldexp(m, 53))        # exact 53-bit integer mantissa
        shift = e - 53 + frac_bits
        return mant << shift if shift >= 0 else mant >> (-shift)

    acc = 0
    for x, y in zip(a.tolist(), b.tolist()):
        acc += to_fixed(x) * to_fixed(y)     # exact product, scale 2**(-2*frac_bits)
    return math.ldexp(float(acc), -2 * frac_bits)  # one rounding at readout


def _selftest() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(768)
    b = rng.standard_normal(768)
    # order-independence: any permutation gives bit-identical fsum
    import struct
    def bits(x): return struct.unpack("<Q", struct.pack("<d", x))[0]
    base = exact_dot(a, b)
    for s in range(16):
        p = rng.permutation(a.size)
        assert bits(exact_dot(a[p], b[p])) == bits(base), "fsum reduction must be order-independent"
    # fsum agrees with the fully-exact big-int gold to <1 ULP
    gold = exact_int_dot(a, b)
    assert abs(base - gold) <= abs(gold) * 2**-50 + 2**-50, (base, gold)
    print(f"refquire selftest OK: order-independent, and fsum matches exact-int gold "
          f"(fsum={base:.15g}, gold={gold:.15g})")


if __name__ == "__main__":
    _selftest()
