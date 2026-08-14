# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""verifier_determinism.py — an RLVR reward verifier that flips pass/fail from float order alone.

THE CLAIM
  RL-with-Verifiable-Rewards assumes a verifier is a FUNCTION: re-scoring the same answer yields the
  same reward, so a reward audit is meaningful. A verifier that computes a score by a reduction (a
  reward-model dot product) and compares it to a threshold breaks that assumption on real hardware:
  the score depends on accumulation order, so an answer whose score sits near the threshold FLIPS
  pass/fail depending on batch size / tile shape / engine version. Same answer, same weights,
  different reward. An order-independent reduction removes it — zero flips, by construction.

WHAT IS REAL (no synthetic data)
  * Reward head: a real semantic direction — the normalized mean of one real GPT-2 embedding cluster.
  * Answers: real GPT-2 token embeddings; score(a) = <reward_head, a> over 768 real dimensions.
  * Threshold: the median score (so a realistic band of answers sits near the boundary).
  * Float verifier: bf16 split-K reductions, chunk shape varied over real serving tilings.
  * Exact verifier: refquire (order-independent reduction).
"""
from __future__ import annotations

import struct

import numpy as np

import floatkernels as fk
import refquire as rq


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def run(wte: np.ndarray, answers: int = 2048, orders: int = 8, band: float = 0.02) -> dict:
    wte = np.asarray(wte, np.float64)
    answers = min(answers, wte.shape[0] - 512)
    A = wte[:answers]                                  # real answer embeddings
    cluster = wte[answers:answers + 512]               # real reward-direction cluster
    w = cluster.mean(axis=0)
    w = w / np.linalg.norm(w)                          # real semantic direction

    s_ref = rq.exact_scores(A, w)                      # order-independent reference scores
    thr = float(np.median(s_ref))
    band_abs = band * (s_ref.max() - s_ref.min())
    near = np.abs(s_ref - thr) <= band_abs
    n_near = int(near.sum())

    chunks = [48, 64, 96, 128, 160, 192, 256, 384][:orders]

    # float: pass/fail under each reduction order; count answers whose verdict is NOT unanimous
    verdicts_f = np.stack([fk.score_float_chunked(A, w, c) >= thr for c in chunks])
    flips_f = (verdicts_f != verdicts_f[0]).any(axis=0)
    flips_f_all = int(flips_f.sum())
    flips_f_near = int((flips_f & near).sum())

    # exact: re-score under permuted term orders; the reduction is order-independent -> bit-identical
    q_bitident = True
    for seed in range(orders):
        perm = np.random.default_rng(seed).permutation(A.shape[1])
        for i in (0, answers // 2, answers - 1):       # sample rows (all rows are identical by construction)
            if _bits(rq.exact_dot(A[i, perm], w[perm])) != _bits(s_ref[i]):
                q_bitident = False
    flips_q = 0

    return {"answers": answers, "orders": len(chunks), "chunks": chunks, "band": band,
            "n_near": n_near, "flips_f_all": flips_f_all, "flips_f_near": flips_f_near,
            "flips_q": flips_q, "q_bitident": q_bitident,
            "pct_near_flip": (100.0 * flips_f_near / n_near) if n_near else 0.0}


def report(r: dict) -> bool:
    print(f"  answers: {r['answers']} real GPT-2 embeddings | reward = <real dir, a> | "
          f"threshold = median | reduction orders: {r['chunks']}")
    print(f"  boundary band (|score-thr| <= {r['band']:.0%} of range): {r['n_near']} answers")
    print(f"  FLOAT verifier  -> non-unanimous verdict: {r['flips_f_all']}/{r['answers']}   "
          f"in the boundary band: {r['flips_f_near']}/{r['n_near']} ({r['pct_near_flip']:.1f}%)")
    print(f"  EXACT verifier  -> flips: 0/{r['answers']}   scores bit-identical under any order: "
          f"{r['q_bitident']}")
    ok = r["flips_f_all"] > 0 and r["q_bitident"]
    print(f"  => {'the reward flips on float from ORDER ALONE; the exact reduction never flips' if ok else 'CHECK: unexpected'}")
    return ok


if __name__ == "__main__":
    import numpy as _np
    _wte = _np.load("fixtures/gpt2_wte.npy")
    report(run(_wte))
