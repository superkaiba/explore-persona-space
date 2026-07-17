#!/usr/bin/env python3
"""Tiny-real CPU e2e fixture for the #1005 driver (#906 recipe, plan §4.7).

Builds a LOCAL model dir pairing the REAL pinned R1-distill TOKENIZER (the
manipulated contract — chat template, think delimiters, bos, all at the pinned
revision) with a from-config TINY same-arch Qwen2 model over the REAL
vocab-id space — so `issue1005_run.py` can execute its ENTIRE production path
(startup asserts → gate → generate[synthetic] → parse → capture → fits) on CPU
with real library types at every internal seam, faking ONLY model scale
(from-config random weights) and the vLLM + Hub boundaries
(``--synthetic-completions`` / ``--no-upload``). The smoke IS the production
driver — no separate smoke path (the unification contract).

Usage::

    uv run python scripts/issue1005_tiny_e2e_fixture.py \\
        --out /tmp/issue-1005-smoke/tiny_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue1005_common import MODEL_REVISION, THINKING_MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="tiny-real #1005 model-dir builder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=THINKING_MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    # REAL tokenizer at the pinned revision (CPU-cheap: tokenizer files only).
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.save_pretrained(out)

    # From-config tiny same-arch model over the REAL vocab-id space (152064 —
    # the profile's token ids must be embeddable; weights are random).
    cfg = Qwen2Config(
        vocab_size=152064,
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 2,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        bos_token_id=151646,
        eos_token_id=151643,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(out)
    print(
        f"[fixture] tiny R1-tokenizer model dir written: {out} "
        f"({args.layers} layers, hidden {args.hidden}, vocab {cfg.vocab_size})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
