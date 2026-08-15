# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""worst_case_order.py — how bad can accumulation ORDER get?

The verifier demo uses realistic bf16 chunk orders and flips ~8-18% of the boundary answers. This asks
the adversarial question: of all the ways to sum the SAME products, which ORDER flips the most verdicts?

An evolutionary search (OpenEvolve driving a code LLM) discovered a principled worst case — sort the
product terms by magnitude, then interleave the largest positive and negative terms, forcing catastrophic
cancellation through the bf16 running sum. On this real GPT-2 reward task it flips ~54% of the boundary
answers. The order-independent reduction flips ZERO no matter the order — the whole point.

Honest scope: this is a worst-case *demonstration* of order-dependence, not a claim any single production
kernel uses this exact order. Real kernels land between the realistic (~8%) and worst-case (~54%) rows.
What's invariant is the bottom row: exact = 0, always.
"""
from __future__ import annotations

import numpy as np

import floatkernels as fk
import refquire as rq


def worst_case_order(products) -> np.ndarray:
    """Return the accumulation order (a permutation of range(len(products))) that maximises bf16 rounding
    error: magnitude-descending, then interleave the largest remaining positive and negative terms."""
    p = np.asarray(products, np.float64)
    idx = np.argsort(np.abs(p))[::-1]                 # magnitude, descending
    pos = idx[p[idx] > 0]
    neg = idx[p[idx] < 0]
    if pos.size == 0 or neg.size == 0:
        return idx
    out, i, j = [], 0, 0
    take_pos = pos.size > 0 and (neg.size == 0 or abs(p[pos[0]]) >= abs(p[neg[0]]))
    while i < pos.size or j < neg.size:
        if take_pos and i < pos.size:
            out.append(pos[i]); i += 1
        elif not take_pos and j < neg.size:
            out.append(neg[j]); j += 1
        take_pos = not take_pos
    return np.asarray(out, dtype=int)


def _bf16_seq_sum(vals) -> float:
    """Sequential bf16 accumulation of `vals` in the given order."""
    acc = np.float32(0.0)
    for v in fk.bf16(vals):
        u = np.float32(acc + v).view(np.uint32)
        acc = ((u + ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF))
               & np.uint32(0xFFFF0000)).view(np.float32)
    return float(acc)


def run(wte, answers: int = 2048, band: float = 0.02) -> dict:
    wte = np.asarray(wte, np.float64)
    answers = min(answers, wte.shape[0] - 512)
    A = wte[:answers]
    w = wte[answers:answers + 512].mean(axis=0)
    w = w / np.linalg.norm(w)
    P = A * w                                          # (N, K) products; each row is one reward score
    s_ref = rq.exact_scores(A, w)                      # exact, order-independent
    thr = float(np.median(s_ref))
    band_abs = band * (s_ref.max() - s_ref.min())
    idx = np.where(np.abs(s_ref - thr) <= band_abs)[0]
    n_b = int(idx.size)
    ref = s_ref[idx] >= thr

    worst = sum((_bf16_seq_sum(P[i][worst_case_order(P[i])]) >= thr) != ref[k]
                for k, i in enumerate(idx))
    chunk = sum((fk.score_float_chunked(P[i][None, :], np.ones(P.shape[1]), 96)[0] >= thr) != ref[k]
                for k, i in enumerate(idx))
    return {"n_boundary": n_b, "worst_flips": int(worst), "chunk_flips": int(chunk),
            "worst_rate": worst / n_b, "chunk_rate": chunk / n_b}


def report(r: dict) -> bool:
    print(f"  boundary answers: {r['n_boundary']}   (same real GPT-2 verifier task)")
    print(f"  realistic bf16 chunk order  -> flips {r['chunk_flips']}/{r['n_boundary']} ({r['chunk_rate']:.0%})")
    print(f"  EVOLVED worst-case order    -> flips {r['worst_flips']}/{r['n_boundary']} ({r['worst_rate']:.0%})"
          "   <- order alone can flip a MAJORITY")
    print(f"  exact quire (any order)     -> flips 0/{r['n_boundary']} (0%)   <- invariant")
    ok = r["worst_flips"] > r["chunk_flips"]
    print(f"  => {'accumulation ORDER alone flips up to ~half the near-threshold verdicts; the exact reduction flips none' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    _wte = np.load("fixtures/gpt2_wte.npy")
    report(run(_wte))
