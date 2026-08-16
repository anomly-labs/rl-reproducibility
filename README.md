<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0. -->
# RL / verifier reproducibility demo

[![CI](https://github.com/anomly-labs/rl-reproducibility/actions/workflows/ci.yml/badge.svg)](https://github.com/anomly-labs/rl-reproducibility/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%20–%203.13-blue.svg)
![deps](https://img.shields.io/badge/deps-numpy%20only-brightgreen.svg)
![runtime](https://img.shields.io/badge/runs%20in-~3s-brightgreen.svg)

```text
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  the SAME reward · the SAME weights · a different (valid) sum order  ║
        ╚══════════════════════════════════════════════════════════════════════╝

             reward(answer)  =   Σ  wᵢ · aᵢ         re-order the sum (same math):
                                  i

               float engine   →   0.50003 PASS      0.49998 FAIL      ← two verdicts
               exact  quire   →   0.50000 PASS      0.50000 PASS      ← one, always

           Same answer, same weights. FLOAT's pass/fail depends on the accumulation
           ORDER, near the threshold — the only place a verdict is ever in doubt.
           The exact quire's never does: bit-identical, every run, zero flips.
```

**Float accumulation *order* alone changes reward verdicts and sampler/trainer probabilities — on real
GPT-2 weights, on your laptop, in seconds. An order-independent reduction removes it, bit-for-bit.**

```mermaid
flowchart LR
    X["same weights<br/>+ inputs"] --> O{"reduction<br/>order"}
    O --> A["float · order A<br/>reward → PASS ✅"]
    O --> B["float · order B<br/>reward → FAIL ❌"]
    O --> C["exact quire · any order<br/>reward → identical ✅"]
    style A fill:#2b2a17,stroke:#c9c15a,color:#ffffff
    style B fill:#3a1414,stroke:#e2554e,color:#ffffff
    style C fill:#12331c,stroke:#3fb56a,color:#ffffff
    style O fill:#1b1b2b,stroke:#8a8ac0,color:#ffffff
```

This is the problem behind the *training–inference mismatch* and *verifier nondeterminism* that RL
post-training teams are patching in software today. It's a property of the arithmetic, not the model.
Anomly puts an order-independent, exact reduction — a 256-bit b-posit **quire** — in silicon at
GPU-parity speed. This repo lets you reproduce the problem, and the class of fix, yourself.

## Quick start

```bash
pip install numpy          # that's the only dependency to run the demo
python demo.py
```

Runs in a few seconds on real GPT-2 embeddings (shipped as fixtures — no download, nothing synthetic).

## What it shows

```
[1] VERIFIER DETERMINISM — does a reward verifier flip pass/fail from float order?
  FLOAT verifier  -> non-unanimous verdict: 30/2048   in the boundary band: 30/170 (17.6%)
  EXACT verifier  -> flips: 0/2048   scores bit-identical under any order: True

[2] TRAINING-INFERENCE MISMATCH — do sampler & trainer disagree from float order?
  FLOAT  -> logits differing: 7492/8192   KL(sampler||trainer): mean 3.17e-02 max 4.97e-02
  EXACT  -> logits bit-identical under any order: 8192/8192   KL(orderA||orderB): 0.0 exactly

[3] RL LOOP FORK — does the same training run diverge just from sampler kernel shape?
  FLOAT  -> action streams identical across chunk: False   final weight max |delta|: 9.33e-03
  EXACT  -> action streams identical across chunk: True   final weights bit-equal: True

[4] WORST-CASE ORDER — how bad can order alone get? (order found by evolutionary search)
  verifier flip-rate:  realistic 12/170 (7%)  -> worst-case 92/170 (54%)  -> exact 0/170 (0%)
  TIM sampler vs exact: realistic KL 2.8e-04  -> worst-case KL 7.8e-02     -> exact 0.0
  (GRPO: the same search finds its training fork already near-saturated under any float order — it forks
   under float and is bit-identical under exact regardless of order; see [3].)

[5] CATASTROPHIC CANCELLATION — is float32 not just non-reproducible, but WRONG?
  a dot product whose TRUE value is 1  -> float32 returns ~1e8-1e9 (off by orders of magnitude)
  exact reduction (any order)          -> the true value, always   (verified vs math.fsum)

[6] NEGATIVE VARIANCE — the naive one-pass variance E[x^2]-E[x]^2 in float32
  large-mean / small-spread data  -> float32 variance goes NEGATIVE (~-1.4e9; std = NaN)
  exact reduction                 -> the correct, non-negative variance, always
```
*(Your numbers will match — the inputs are fixed and disclosed. Whole suite runs in ~3s.)*

- **Verifier determinism** (`verifier_determinism.py`) — a reward-model score is a dot product compared
  to a threshold. Compute it in different bf16 reduction orders (the shapes that differ between serving
  configs) and answers near the boundary **flip pass/fail** — same answer, same weights, different
  reward. The order-independent reduction never flips.
- **Training–inference mismatch** (`tim.py`) — the trainer (sequential) and sampler (chunked-tree)
  accumulate the *same* GPT-2 logits in different orders and assign **different probabilities to the same
  tokens** (KL > 0). The order-independent reduction is bit-identical, KL = 0.
- **RL loop fork** (`grpo.py`) — a real GRPO loop (verifiable nearest-neighbour reward, reward learns
  0→0.48) run twice with identical seeds and data, changing **only** the sampler's split-K chunk shape.
  Under float the two runs **fork** — different sampled action streams, different trained weights — so
  you can't reproduce the run. Under the order-independent reduction they're **bit-identical** end to end.
  (Honest scope: at this scale it's a training-*reproducibility* failure; the behavioral pass/fail
  consequence is demo [1].)
- **Worst-case order** (`worst_case_order.py`) — realistic kernels flip ~7-18%, but *how bad can order
  alone get?* An evolutionary search (OpenEvolve driving a code LLM) discovered a principled worst case —
  sort by magnitude, then interleave the largest positive and negative terms to force catastrophic
  cancellation in the bf16 sum. On the **verifier** it flips **~54%** of the boundary verdicts (vs ~7%
  realistic); on the **TIM sampler** it pushes the token distribution **~276× further** from the exact
  ground truth than a normal order does (KL 7.8e-2 vs 2.8e-4). The same search independently rediscovered
  the *same* ordering principle for each demo. The exact reduction is **0** in every case, no matter the
  order. (Honest scope: a worst-case *demonstration* of order-dependence, not a claim a production kernel
  uses this exact order; real kernels land between the realistic and worst-case rows. On the **GRPO**
  training loop the same search found the fork already near-saturated under any float order — it forks
  under float and is bit-identical under exact regardless — so there's no distinct worst-case row for it.)
- **Catastrophic cancellation** (`worst_case_cancellation.py`) — the other face of the problem: not just
  that order changes the answer, but that float32 can be flatly **WRONG**. A dot product of ordinary
  float-representable numbers whose true value is **1** (thousands of large ± terms plus a residual of 1)
  returns **~1e8–1e9** in float32 — off by orders of magnitude — while the order-independent exact
  reduction returns exactly 1 (verified vs `math.fsum`). (Honest scope: a constructed worst case, to bound
  how wrong float gets on ordinary inputs; it's a *correctness* statement, not only a reproducibility one.)
- **Negative variance** (`worst_case_variance.py`) — the textbook one-pass variance `E[x²] − E[x]²`, used
  in countless naive stats/ML routines, on large-mean / small-spread data returns a **NEGATIVE** float32
  variance (~−1.4e9 — mathematically impossible; `std = √negative = NaN`, whitening/Cholesky blows up)
  where the true variance is ~0. The exact reduction computes the identical formula correctly and stays
  non-negative. (Honest scope: constructed worst-case data; inputs are snapped to float32 so the only
  error measured is the accumulation.)

## What's real, and what's honest scope

- **Real data:** real GPT-2 `wte` / `wpe` rows from the published `openai-community/gpt2` checkpoint
  (`regenerate_fixtures.py` rebuilds them). No synthetic numbers.
- **The float kernels** (`floatkernels.py`) are faithful reference implementations of the two real kernel
  families (trainer sequential vs sampler split-K tree), in bf16 — only the accumulation order differs.
- **The exact path** (`refquire.py`) is an *order-independent reference reduction* (correctly-rounded
  `math.fsum`, cross-checked against an arbitrary-precision big-int gold — run `python refquire.py`). It
  demonstrates the **property**: an order-independent accumulator eliminates the nondeterminism.
- **Anomly's silicon goes further and faster.** A 256-bit b-posit quire keeps the *products* exact too
  and runs at GPU-parity throughput — measured **131,072 / 131,072 outputs bit-identical** on our Alveo
  U200 engine, with a reward verifier's flip-rate driven to **zero**. This repo proves the class of fix;
  the fast, fully-exact hardware datapath is Anomly's.

We certify **reproducibility** — the same defined computation, bit-identical anywhere. For the reductions
here (dot products, variance moments), the exact path is also **correct** — it returns the correctly-rounded
true value, verified against arbitrary precision (demos [5]-[6]). We do not claim exactness of transcendentals
(`sin`, `exp`, …); the claim is exact, order-independent *accumulation*.

## Files

| file | what |
|---|---|
| `demo.py` | runs all six demonstrations, one consolidated verdict |
| `verifier_determinism.py` | reward-verifier flip demo |
| `tim.py` | training–inference mismatch demo |
| `grpo.py` | RL-loop-fork demo (training run diverges under float) |
| `worst_case_order.py` | evolutionary-search worst-case accumulation order (~54% flips vs 0%) |
| `worst_case_cancellation.py` | catastrophic cancellation — float32 dot ~1e9 when the truth is 1 |
| `worst_case_variance.py` | naive one-pass variance goes NEGATIVE in float32 |
| `refquire.py` | order-independent exact reference reduction (+ big-int gold check) |
| `floatkernels.py` | the order-dependent bf16 serving/training kernels |
| `regenerate_fixtures.py` | rebuild the real GPT-2 fixtures from the public checkpoint |

## About Anomly

Anomly builds AI compute that returns **exact, reproducible answers by default** — a b-posit number
format with a 256-bit quire, so accumulation is order-independent and bit-identical across hardware.
Silicon-proven (a 625-core engine on an Alveo U200 FPGA; a first SKY130 tapeout; running on Tenstorrent
Blackhole), formally verified, and validated by John Gustafson — inventor of the posit/quire arithmetic.

If reproducible RL, stable verifiers, or auditable inference matter to you, we'd love to run a short
pilot measuring the flip-rate on *your* models and taking it to zero.

**Anomly, Inc. · https://www.anomly.com/contact**

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). This repo is a demonstration; the Apache-2.0 patent grant
covers only the code here, which contains none of Anomly's b-posit / quire silicon IP or patents.
