#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""demo.py — the ~3-minute reproducibility demo for RL / verifier teams.

Runs both demonstrations back-to-back on REAL GPT-2 weights and prints one consolidated verdict:

  1. VERIFIER DETERMINISM — a reward verifier flips pass/fail on a measurable fraction of boundary
     answers from float accumulation ORDER alone; an order-independent reduction never flips.
  2. TRAINING-INFERENCE MISMATCH (TIM) — the sampler and trainer assign different probabilities to
     the same tokens from order alone; the order-independent reduction is bit-identical, KL = 0.

Nothing synthetic — every number is computed live from the real GPT-2 embedding fixtures.

    python demo.py            # uses fixtures/gpt2_wte.npy + fixtures/gpt2_wpe.npy

The exact path here is an order-independent REFERENCE reduction (refquire, correctly-rounded sum).
Anomly's silicon does the same thing fully-exact and at GPU-parity speed — a 256-bit b-posit quire,
measured 131,072/131,072 outputs bit-identical on our Alveo U200 engine, and a reward verifier's
flip-rate driven to zero. This repo lets you reproduce the *problem*, and the *class of fix*, yourself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import tim
import verifier_determinism as vd

HERE = Path(__file__).resolve().parent


def main() -> int:
    wte_p = HERE / "fixtures" / "gpt2_wte.npy"
    wpe_p = HERE / "fixtures" / "gpt2_wpe.npy"
    if not wte_p.exists() or not wpe_p.exists():
        print("Missing fixtures. Run:  python regenerate_fixtures.py   (needs torch + transformers)")
        return 2
    wte = np.load(wte_p)
    wpe = np.load(wpe_p)

    bar = "=" * 82
    print(bar)
    print("Anomly — RL / verifier reproducibility demo  (real GPT-2 weights, on your machine)")
    print(bar)
    t0 = time.perf_counter()

    print("\n[1] VERIFIER DETERMINISM — does a reward verifier flip pass/fail from float order?")
    ok1 = vd.report(vd.run(wte))

    print("\n[2] TRAINING-INFERENCE MISMATCH — do sampler & trainer disagree from float order?")
    ok2 = tim.report(tim.run(wte, wpe))

    dt = time.perf_counter() - t0
    print("\n" + bar)
    ok = ok1 and ok2
    if ok:
        print("RESULT: on real GPT-2 data, float accumulation ORDER alone changes reward verdicts and")
        print("        sampler/trainer probabilities. An order-independent reduction removes both,")
        print("        bit-for-bit. Anomly puts that reduction in silicon at GPU-parity speed.")
    else:
        print("RESULT: a property did not hold — investigate before using this externally.")
    print(f"[{dt:.1f}s]   Contact: https://www.anomly.com/contact")
    print(bar)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
