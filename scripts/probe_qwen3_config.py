#!/usr/bin/env python3
"""Issue #506 Phase-0a item 4 — Qwen3-32B config probe.

CPU-only. Downloads ``Qwen/Qwen3-32B``'s config.json from HF Hub (a few KB),
asserts the architectural invariants that downstream code depends on:

  - ``num_attention_heads == 64``
  - ``num_key_value_heads == 8``         (vLLM TP ∈ {1, 2, 4, 8} all legal)
  - ``num_hidden_layers == 64``
  - ``hidden_size == 5120``

Also re-tokenizes the marker / trigger to confirm ``EXPECTED_MARKER_ID``
(83399 — the canonical Qwen-family leading-space ※; same id as Qwen-2.5-7B)
and vocab sizes.

Writes ``eval_results/issue_506/qwen3_config_probe.json`` with the probed
values so the dispatcher's pre-launch sanity check can read them at runtime.

(Renamed from ``probe_qwen35_config.py`` at round 5 alongside the
``Qwen3.5-27B`` → ``Qwen3-32B`` model swap. Plan v3 §4.3.1 item 4 is
otherwise unchanged.)

Usage:
    uv run python scripts/probe_qwen3_config.py [--strict-vocab]

PASS exit 0; FAIL (head-count drift, marker drift) exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="probe_qwen3_config")

from _issue506_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_RESULTS_DIR,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_ATTENTION_HEADS,
    EXPECTED_NUM_HIDDEN_LAYERS,
    EXPECTED_NUM_KEY_VALUE_HEADS,
    marker_preflight,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe Qwen3-32B config.json + tokenizer for #506 invariants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--strict-vocab",
        action="store_true",
        help="Treat tokenizer vocab-size mismatches as FAIL (default WARN).",
    )
    return p.parse_args()


def _resolve_head_counts(cfg) -> tuple[int, int, int, int]:
    """Pull (num_attention_heads, num_key_value_heads, num_hidden_layers, hidden_size).

    Qwen3-32B is a standard dense causal-LM (``Qwen3ForCausalLM``,
    ``model_type: qwen3``) with attributes at the top level — no
    ``text_config`` nesting like the abandoned Qwen3.5-27B multimodal had.
    The resolver still checks ``text_config`` first for forward-compat
    with future configs.
    """
    text_cfg = getattr(cfg, "text_config", None)
    src = text_cfg if text_cfg is not None else cfg
    return (
        int(src.num_attention_heads),
        int(src.num_key_value_heads),
        int(src.num_hidden_layers),
        int(src.hidden_size),
    )


def main() -> int:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    # Read raw config.json directly. (This pattern was originally adopted
    # to skirt the ``KeyError: 'qwen3_5'`` on the abandoned Qwen3.5-27B
    # multimodal config; ``Qwen3-32B``'s ``model_type: qwen3`` IS registered
    # by the pinned transformers and ``AutoConfig.from_pretrained`` would
    # work too. Kept JSON-only for CPU-light, side-effect-free probing.)
    raw_path = hf_hub_download(
        BASE_MODEL,
        "config.json",
        token=os.environ.get("HF_TOKEN"),
    )
    raw_cfg = json.loads(Path(raw_path).read_text())

    # Wrap raw_cfg as a SimpleNamespace tree so the head-count resolver works
    # uniformly with both AutoConfig objects (pod path) and dict-loaded JSON
    # (host path).
    from types import SimpleNamespace

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    cfg = _to_ns(raw_cfg)
    nah, nkvh, nhl, hs = _resolve_head_counts(cfg)

    print(f"Probed {BASE_MODEL}:")
    print(f"  num_attention_heads = {nah} (expected {EXPECTED_NUM_ATTENTION_HEADS})")
    print(f"  num_key_value_heads = {nkvh} (expected {EXPECTED_NUM_KEY_VALUE_HEADS})")
    print(f"  num_hidden_layers   = {nhl} (expected {EXPECTED_NUM_HIDDEN_LAYERS})")
    print(f"  hidden_size         = {hs} (expected {EXPECTED_HIDDEN_SIZE})")

    failed: list[str] = []
    if nah != EXPECTED_NUM_ATTENTION_HEADS:
        failed.append(f"num_attention_heads: {nah} != {EXPECTED_NUM_ATTENTION_HEADS}")
    if nkvh != EXPECTED_NUM_KEY_VALUE_HEADS:
        failed.append(f"num_key_value_heads: {nkvh} != {EXPECTED_NUM_KEY_VALUE_HEADS}")
    if nhl != EXPECTED_NUM_HIDDEN_LAYERS:
        failed.append(f"num_hidden_layers: {nhl} != {EXPECTED_NUM_HIDDEN_LAYERS}")
    if hs != EXPECTED_HIDDEN_SIZE:
        failed.append(f"hidden_size: {hs} != {EXPECTED_HIDDEN_SIZE}")
    if failed:
        print("\nFAIL — Qwen3-32B architectural drift:")
        for line in failed:
            print(f"  - {line}")
        print("Re-plan: vLLM TP choice + memory budget depend on these.")
        return 1

    # Marker / tokenizer preflight (re-uses #475 logic).
    preflight = marker_preflight(require_strict_vocab=args.strict_vocab)

    out_dir = EVAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qwen3_config_probe.json"
    payload = {
        "base_model": BASE_MODEL,
        "num_attention_heads": nah,
        "num_key_value_heads": nkvh,
        "num_hidden_layers": nhl,
        "hidden_size": hs,
        **preflight,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nOK: wrote probe to {out_path}")
    print(f"OK: marker id={preflight['marker_ids']}, trigger ids={preflight['trigger_ids']}")
    print(
        "Note: vLLM TP must divide BOTH num_attention_heads (64) AND num_key_value_heads (8); "
        "legal TP ∈ {1, 2, 4, 8}. Default TP=8 per plan v4 §11.2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
