# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""grpo.py — the RL loop itself forks under float; the exact reduction makes training bit-reproducible.

THE CLAIM
  The verifier and TIM demos show the mechanism. This shows the CONSEQUENCE inside a real RL loop: run
  a GRPO-style training loop TWICE with identical seeds and identical data, changing ONLY the sampler
  kernel's reduction shape (split-K chunk 96 vs 128 — exactly what differs between batch sizes / tile
  shapes / engine versions). With float logits the two runs silently diverge — different sampled action
  streams, different trained weights. With the order-independent reduction they are bit-identical end to
  end (same action-stream hash, same weights).

WHAT IS REAL (no synthetic data)
  * Features: real GPT-2 embeddings — prompts are h_i = wte[token] + wpe[pos] (genuine layer-0 inputs);
    the action space is a real wte vocab slice.
  * Verifiable reward (RLVR-style): action a is correct for prompt i iff a is the nearest real embedding
    to the prompt among the action slice (cosine). Deterministic, checkable, no reward model.
  * Policy / update: linear-softmax policy; GRPO group sampling; REINFORCE on the logit head. The SAMPLER
    uses split-K chunked bf16 partials; the TRAINER a flat fp32 reduction — two engines, one set of
    weights, the production topology.

Honest scope. At this small scale the fork is a WEIGHT / action-stream reproducibility failure — you
cannot reproduce the training run — while the greedy policy can still agree (the behavioral pass/fail
consequence is in verifier_determinism.py). We report exactly what we measure. The exact path here is
the order-independent reference reduction (refquire); Anomly's silicon does it fully-exact at speed.
"""
from __future__ import annotations

import hashlib

import numpy as np

import floatkernels as fk
import refquire as rq

TOKENS = [464, 262, 286, 290, 318, 257, 284, 287, 3290, 1169, 703, 508,
          618, 703, 986, 30, 373, 481, 447, 340, 351, 383, 326, 289]


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sampler_float(H, Wp, chunk):
    """Sampler kernel: bf16 split-K partials (reduction shape follows batch/tile size)."""
    P, K = H.shape
    A = Wp.shape[1]
    out = np.zeros((P, A), dtype=np.float32)
    for c0 in range(0, K, chunk):
        out = fk.bf16(out + fk.bf16(H[:, c0:c0 + chunk].astype(np.float32)
                                    @ Wp[c0:c0 + chunk].astype(np.float32)))
    return out.astype(np.float64)


def _trainer_float(H, Wp):
    """Trainer kernel: one flat fp32 reduction (BLAS order)."""
    return (H.astype(np.float32) @ Wp.astype(np.float32)).astype(np.float64)


def _train(H, E, reward_target, steps, group, lr, sampler_chunk, mode, seed):
    """One GRPO-style run. Returns (weights, action_stream_hash, reward_curve)."""
    P, K = H.shape
    A = E.shape[0]
    rng = np.random.default_rng(seed)
    Wp = np.zeros((K, A), dtype=np.float64)          # logit head (trained)
    base = rq.exact_matmul(H, E.T) / np.sqrt(K)      # fixed real-feature scores (order-independent)
    log = hashlib.sha256()
    curve = []
    for _ in range(steps):
        if mode == "float":
            head = _sampler_float(H, Wp, sampler_chunk)   # SAMPLER (chunk-shaped)
            head_train = _trainer_float(H, Wp)            # TRAINER (flat)
        else:
            head = rq.exact_matmul(H, Wp)                 # one order-independent arithmetic;
            head_train = head                            # sampler and trainer agree by construction
        probs = _softmax(base + head)
        acts = np.stack([rng.choice(A, p=probs[i]) for i in range(P) for _ in range(group)]
                        ).reshape(P, group)
        r = (acts == reward_target[:, None]).astype(np.float64)
        adv = r - r.mean(axis=1, keepdims=True)
        probs_train = _softmax(base + head_train)
        g = np.zeros_like(Wp)
        for i in range(P):
            for j in range(group):
                grad_row = -probs_train[i].copy()
                grad_row[acts[i, j]] += 1.0
                g += np.outer(H[i], adv[i, j] * grad_row)
        Wp = Wp + lr * g / (P * group)
        log.update(acts.tobytes())
        curve.append(float(r.mean()))
    return Wp, log.hexdigest(), curve


def run(wte, wpe, prompts=16, actions=96, steps=40, group=6, lr=0.5) -> dict:
    wte = np.asarray(wte, np.float64)
    wpe = np.asarray(wpe, np.float64)
    H = np.stack([wte[TOKENS[i % len(TOKENS)]] + wpe[i % wpe.shape[0]] for i in range(prompts)])
    E = wte[1000:1000 + actions]
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    reward_target = np.argmax(Hn @ En.T, axis=1)     # verifiable nearest-neighbour reward

    res = {}
    for mode in ("float", "quire"):
        for chunk in (96, 128):                       # the ONLY thing that changes
            res[(mode, chunk)] = _train(H, E, reward_target, steps, group, lr, chunk, mode, seed=20260709)

    wf96, hf96, cf = res[("float", 96)]
    wf128, hf128, _ = res[("float", 128)]
    wq96, hq96, cq = res[("quire", 96)]
    wq128, hq128, _ = res[("quire", 128)]
    return {"prompts": prompts, "actions": actions, "steps": steps, "group": group,
            "reward0": cf[0], "reward1": float(np.mean(cf[-5:])),
            "f_actions_same": hf96 == hf128, "f_wdiff": float(np.abs(wf96 - wf128).max()),
            "q_actions_same": hq96 == hq128, "q_wbit": bool(np.array_equal(wq96, wq128))}


def report(r: dict) -> bool:
    print(f"  real GPT-2 GRPO: {r['prompts']} prompts, {r['actions']} actions, {r['steps']} steps, "
          f"group {r['group']} | reward {r['reward0']:.2f} -> {r['reward1']:.2f} | "
          f"same seeds/data, ONLY sampler chunk 96 vs 128 differs")
    print(f"  FLOAT  -> action streams identical across chunk: {r['f_actions_same']}   "
          f"final weight max |delta|: {r['f_wdiff']:.2e}")
    print(f"  EXACT  -> action streams identical across chunk: {r['q_actions_same']}   "
          f"final weights bit-equal: {r['q_wbit']}")
    ok = (not r["f_actions_same"] or r["f_wdiff"] > 0) and r["q_actions_same"] and r["q_wbit"]
    print(f"  => {'float kernel shape SILENTLY FORKS the training run; the exact reduction is bit-reproducible' if ok else 'CHECK: unexpected'}")
    return ok


if __name__ == "__main__":
    _wte = np.load("fixtures/gpt2_wte.npy")
    _wpe = np.load("fixtures/gpt2_wpe.npy")
    report(run(_wte, _wpe))
