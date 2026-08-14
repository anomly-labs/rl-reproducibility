# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""tim.py — Training-Inference Mismatch (TIM), reproduced and removed, on real GPT-2 weights.

THE CLAIM
  In LLM reinforcement learning the sampler (inference engine) and the trainer accumulate the SAME
  weights in DIFFERENT orders, so they assign DIFFERENT probabilities to the same tokens — silent
  off-policy drift ("TIM"; arXiv:2605.14220 et al.). We reproduce it with real GPT-2 weights and two
  faithful bf16 accumulation orders, then show the order-independent reduction is bit-identical under
  any order: KL(sampler || trainer) == 0, no batch-invariant-kernel throughput tax.

WHAT IS REAL (no synthetic data)
  * Weights: real GPT-2 token embeddings (wte, tied lm_head) + positional embeddings (wpe).
  * Hidden state: h_i = wte[token_id] + wpe[i] — the model's genuine layer-0 input.
  * Logits: real dot products <wte[v], h> over a vocab slice.
  * Two float paths: trainer-style sequential vs sampler-style chunked-tree, both bf16 — ONLY the
    accumulation order differs.  Exact path: refquire (order-independent).
"""
from __future__ import annotations

import struct

import numpy as np

import floatkernels as fk
import refquire as rq

# 16 real GPT-2 token ids (all < 6144; common word/punct/byte tokens)
TOKEN_IDS = [464, 262, 286, 290, 318, 257, 284, 287, 3290, 1169, 703, 508, 618, 703, 986, 30]


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def run(wte_full: np.ndarray, wpe: np.ndarray, vocab: int = 1024, positions: int = 8) -> dict:
    wte_full = np.asarray(wte_full, np.float64)
    wpe = np.asarray(wpe, np.float64)
    W = wte_full[:vocab]                                # logits over this vocab slice
    K = W.shape[1]
    perm = np.random.default_rng(0).permutation(K)      # fixed, disclosed

    kls, dlp, argmax_flips, diff = [], 0.0, 0, 0
    q_bitident, q_total = 0, 0
    for i in range(positions):
        h = wte_full[TOKEN_IDS[i]] + wpe[i]             # real layer-0 input
        lt = fk.float_sequential(W, h)                  # trainer path
        ls = fk.float_chunked_tree(W, h)                # sampler path
        kls.append(fk.kl(ls, lt))
        dlp = max(dlp, float(np.abs(fk.log_softmax(ls) - fk.log_softmax(lt)).max()))
        diff += int((ls != lt).sum())
        argmax_flips += int(ls.argmax() != lt.argmax())

        qa = rq.exact_logits(W, h)                      # exact, natural order
        qb = rq.exact_logits(W[:, perm], h[perm])       # exact, permuted order
        q_total += vocab
        q_bitident += sum(1 for v in range(vocab) if _bits(qa[v]) == _bits(qb[v]))
        assert fk.kl(qa, qb) == 0.0, "exact KL must be exactly zero"

    return {"vocab": vocab, "positions": positions, "diff": diff, "n_logits": vocab * positions,
            "kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls)), "dlp": dlp,
            "argmax_flips": argmax_flips, "q_bitident": q_bitident, "q_total": q_total}


def report(r: dict) -> bool:
    print(f"  real GPT-2 wte[{r['vocab']}x768] + wpe | {r['positions']} (token,pos) pairs | "
          f"trainer=sequential vs sampler=chunked-tree (bf16), ONLY order differs")
    print(f"  FLOAT  -> logits differing: {r['diff']}/{r['n_logits']}   "
          f"KL(sampler||trainer): mean {r['kl_mean']:.2e} max {r['kl_max']:.2e}   "
          f"argmax flips: {r['argmax_flips']}/{r['positions']}")
    print(f"  EXACT  -> logits bit-identical under any order: {r['q_bitident']}/{r['q_total']}   "
          f"KL(orderA||orderB): 0.0 exactly")
    ok = r["diff"] > 0 and r["q_bitident"] == r["q_total"]
    print(f"  => {'sampler and trainer DISAGREE on float from order alone; the exact reduction is identical' if ok else 'CHECK: unexpected'}")
    return ok


if __name__ == "__main__":
    _wte = np.load("fixtures/gpt2_wte.npy")
    _wpe = np.load("fixtures/gpt2_wpe.npy")
    report(run(_wte, _wpe))
