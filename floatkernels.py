# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""floatkernels.py — the float reductions real serving/training engines actually use.

These are reference implementations of two REAL kernel families that produce the SAME math in
DIFFERENT accumulation orders — exactly what differs between a training engine and an inference
engine, or between two tile/batch configurations of the same engine:

  * float_sequential  — trainer-style left-to-right accumulation
  * float_chunked_tree — sampler-style split-K partials combined pairwise-tree
  * score_float_chunked — reward-model score by bf16 split-K, chunk-shaped

All accumulate in bfloat16 (the aggressive-but-common case). Same weights, same inputs, same dtype —
only the order differs, and that is enough to change the result. Compare with refquire.exact_* which
does not change.
"""
from __future__ import annotations

import numpy as np


def bf16(x) -> np.ndarray:
    """Round a float array to bfloat16 (round-to-nearest-even), returned as float32."""
    u = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def score_float_chunked(A, w, chunk: int) -> np.ndarray:
    """Reward-model score by bf16 split-K reduction, chunk-shaped (the serving fast path;
    reduction shape follows batch/tile size)."""
    A = np.asarray(A, np.float32)
    w = np.asarray(w, np.float32)
    N, K = A.shape
    acc = np.zeros(N, dtype=np.float32)
    for c0 in range(0, K, chunk):
        acc = bf16(acc + bf16(A[:, c0:c0 + chunk] @ w[c0:c0 + chunk]))
    return acc.astype(np.float64)


def float_sequential(W, h) -> np.ndarray:
    """Trainer-style: left-to-right accumulation in bf16."""
    W = np.asarray(W, np.float32)
    h = np.asarray(h, np.float32)
    prods = bf16(W * h)                       # (V, K)
    acc = np.zeros(W.shape[0], dtype=np.float32)
    for k in range(prods.shape[1]):
        acc = bf16(acc + prods[:, k])
    return acc.astype(np.float64)


def float_chunked_tree(W, h, chunk: int = 96) -> np.ndarray:
    """Sampler-style: split-K chunked partials, combined pairwise-tree, in bf16."""
    W = np.asarray(W, np.float32)
    h = np.asarray(h, np.float32)
    V, K = W.shape
    prods = bf16(W * h)
    parts = []
    for c0 in range(0, K, chunk):
        acc = np.zeros(V, dtype=np.float32)
        for k in range(c0, min(c0 + chunk, K)):
            acc = bf16(acc + prods[:, k])
        parts.append(acc)
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts) - 1, 2):
            nxt.append(bf16(parts[i] + parts[i + 1]))
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0].astype(np.float64)


def log_softmax(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    z = x - x.max()
    return z - np.log(np.exp(z).sum())


def kl(p_logits, q_logits) -> float:
    lp = log_softmax(p_logits)
    lq = log_softmax(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum())
