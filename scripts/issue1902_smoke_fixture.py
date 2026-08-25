#!/usr/bin/env python3
"""Issue #1902 VM smoke fixture: tiny-real Olmo2/Olmo3 model + smoke corpus.

Extends the unit-B test fixture (2-layer random-weights Olmo2 over the REAL
vocab) into the reusable full-chain smoke helper (#906 tiny-real standard):

1. ``--model-dir``: saves an 8-layer h=64 random-weights ``Olmo2ForCausalLM``
   (default) or — ``--arch olmo3``, the #2544 widening promised by that
   plan's smoke enumeration — an ``Olmo3ForCausalLM`` whose DEFAULT
   ``layer_types`` at 8 layers puts full attention at {3, 7} (the production
   ``i % 4 == 3`` pattern at tiny depth) with ``sliding_window=4096``
   matching ``OLMO3_SLIDING_WINDOW``; plus the REAL tokenizer for the arch
   (real BPE ids) — the ONE sanctioned fake is GPU-scale weights. 8 layers
   (not 2) so the depth-relative probe-layer rule resolves in-range indices.
2. ``--corpus-dir``: builds the smoke corpus from a REAL ``--probe`` output
   (real LMSYS rows — text untouched, digest-only handling): relabels the
   probe's single ``unclustered`` group into round-robin ``cluster_k`` groups
   (the production fold-structure shape) and REORDERS the marked gsm8k/mbpp
   strata rows inside the smoke slice window so the stratum arm class stays
   smoke-covered. Group labels are fold-structure metadata, never row text —
   the relabel is a disclosed smoke-fixture preparation. ``--probe-dir`` is
   OPTIONAL: without it only the tiny model is built (#2544 stages its real
   corpus from the production HF prefix, so no fixture corpus exists there).

Usage::

    uv run python scripts/issue1902_smoke_fixture.py \
        --probe-dir /tmp/issue1902_probe --out /tmp/issue1902_smoke
    uv run python scripts/issue1902_smoke_fixture.py \
        --arch olmo3 --out /tmp/issue2544_smoke_fixture   # model only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1902_common as C  # noqa: E402

TINY_LAYERS = 8
TINY_HIDDEN = 64
SMOKE_WINDOW = 32  # issue1902_run.SMOKE_SINGLE_N — strata must land inside it
N_SMOKE_GROUPS = 8


ARCH_TOKENIZER = {
    "olmo2": "allenai/OLMo-2-1124-7B-Instruct",
    # Instruct tokenizer (same vocab as the base): the fixture serves EVERY
    # ladder rung under smoke, and #2544's natgen cells assert
    # has_chat_template on post rungs (R) — a template-less base tokenizer
    # kills pass1 there (measured 2026-08-24). Mirrors the olmo2 row's
    # Instruct choice; plain-render cells ignore the template.
    "olmo3": "allenai/Olmo-3-7B-Instruct",  # rung R's production id (ladder JSON)
}


def build_tiny_model(model_dir: Path, arch: str = "olmo2") -> None:
    """Random-weights 8-layer Olmo2/Olmo3 over the REAL vocab + REAL tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo2Config, Olmo3Config

    tok = AutoTokenizer.from_pretrained(ARCH_TOKENIZER[arch])
    config_cls = Olmo2Config if arch == "olmo2" else Olmo3Config
    cfg = config_cls(
        vocab_size=len(tok),
        hidden_size=TINY_HIDDEN,
        intermediate_size=2 * TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    if arch == "olmo3":
        # Default Olmo3 layer_types at 8 layers = full attention at {3, 7}
        # (the production i % 4 == 3 pattern); assert rather than assume.
        full = [k for k, t in enumerate(cfg.layer_types) if t == "full_attention"]
        assert full == [3, 7], f"unexpected tiny-olmo3 layer_types: {cfg.layer_types}"
        assert cfg.sliding_window == 4096, cfg.sliding_window
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    assert (model_dir / "config.json").exists()
    print(f"[fixture] tiny {arch} saved -> {model_dir} (layers={TINY_LAYERS}, h={TINY_HIDDEN})")


def build_smoke_corpus(probe_dir: Path, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for name in (C.CORPUS_SINGLE_FILENAME, C.CORPUS_MULTI_FILENAME):
        rows = [
            json.loads(line)
            for line in (probe_dir / name).read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        strata = [r for r in rows if r["class"] in (C.CLASS_GSM8K, C.CLASS_MBPP)]
        generic = [r for r in rows if r["class"] not in (C.CLASS_GSM8K, C.CLASS_MBPP)]
        # strata inside the smoke slice window (arm-class coverage)
        reordered = generic[: SMOKE_WINDOW - len(strata)] + strata
        reordered += generic[SMOKE_WINDOW - len(strata) :]
        for k, r in enumerate(reordered):
            if r["class"] in (C.CLASS_GSM8K, C.CLASS_MBPP):
                pass  # marked strata keep their own whole-stratum group labels
            else:
                r["cluster"] = k % N_SMOKE_GROUPS
                r["group"] = f"cluster_{k % N_SMOKE_GROUPS}"
        with open(corpus_dir / name, "w", encoding="utf-8") as f:
            for r in reordered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[fixture] {name}: {len(reordered)} rows ({len(strata)} strata in-window)")
    for extra in (C.CLUSTERS_FILENAME, "manifest_stats.json"):
        if (probe_dir / extra).exists():
            shutil.copy(probe_dir / extra, corpus_dir / extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", choices=sorted(ARCH_TOKENIZER), default="olmo2")
    args = ap.parse_args()
    build_tiny_model(args.out / "model", arch=args.arch)
    if args.probe_dir is not None:
        build_smoke_corpus(args.probe_dir, args.out / "corpus")
    else:
        print("[fixture] no --probe-dir: model only (corpus staged elsewhere)", flush=True)
    print("[fixture] done", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
