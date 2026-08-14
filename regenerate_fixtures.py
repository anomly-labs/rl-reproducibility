#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed under the Apache License, Version 2.0.
"""regenerate_fixtures.py — re-extract the real GPT-2 embedding fixtures (transparency).

The demo ships fixtures/gpt2_wte.npy and fixtures/gpt2_wpe.npy so it runs with numpy alone. Those are
real rows of the *published* openai-community/gpt2 checkpoint — nothing synthetic. This script rebuilds
them from scratch so anyone can confirm provenance.

    pip install torch transformers
    python regenerate_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"
ROWS_WTE = 6144   # >= answers(2048)+cluster(512) and >= vocab(1024); a small real slice
ROWS_WPE = 64


def main() -> int:
    from transformers import GPT2Model  # noqa: import-time dependency, by design
    m = GPT2Model.from_pretrained("gpt2")
    wte = m.wte.weight.detach().float().numpy()   # (50257, 768) real token embeddings (tied lm_head)
    wpe = m.wpe.weight.detach().float().numpy()   # (1024, 768)  real positional embeddings
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "gpt2_wte.npy", np.ascontiguousarray(wte[:ROWS_WTE]).astype(np.float32))
    np.save(OUT / "gpt2_wpe.npy", np.ascontiguousarray(wpe[:ROWS_WPE]).astype(np.float32))
    print(f"wrote {OUT}/gpt2_wte.npy {wte[:ROWS_WTE].shape} and gpt2_wpe.npy {wpe[:ROWS_WPE].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
