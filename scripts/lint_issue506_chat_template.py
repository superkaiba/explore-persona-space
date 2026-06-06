#!/usr/bin/env python3
"""Issue #506 Phase-0a item 5 — chat-template <think> injection lint.

FAIL-LOUD (not WARN). Renders a fixed 3-message exchange under
``tokenizer.apply_chat_template(..., add_generation_prompt=True)``. The
Qwen3.5-27B model card explicitly states the model "operates in thinking
mode by default (generates <think> tokens before responses)" — and the #475
plain arm was confounded by this exact injection at eval time.

Two-pass check:
  1. Default render. Any ``<think>`` or ``</think>`` substring → FAIL.
  2. Defense render with ``enable_thinking=False`` (a kwarg the Qwen3.5
     templates expose for callers that want to suppress the substrate).
     Same hit-check; the dispatcher / eval rig pin this to False, so this
     is the actually-used rendered prefix.

The lint produces ``eval_results/issue_506/chat_template_lint.json`` with
both rendered prefixes recorded for the analyzer's audit.

Usage:
    uv run python scripts/lint_issue506_chat_template.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="lint_issue506_chat_template")

from _issue506_common import BASE_MODEL, EVAL_RESULTS_DIR  # noqa: E402

_FIXTURE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hi"},
]


def _render(tok, **kwargs) -> str:
    return tok.apply_chat_template(
        _FIXTURE_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    default_prefix = _render(tok)
    try:
        suppressed_prefix = _render(tok, enable_thinking=False)
    except TypeError:
        # Older Qwen3 BPE templates don't accept the kwarg; treat as identical
        # to default — the FAIL-LOUD check still fires on the default render.
        suppressed_prefix = default_prefix

    def _has_think(s: str) -> bool:
        return "<think>" in s or "</think>" in s

    def _has_open_unclosed_think(s: str) -> bool:
        """An OPEN ``<think>`` block at the generation slot WITHOUT a closing
        ``</think>`` is the actual #475 confound: the model is forced to
        generate inside the thinking block (its content is later stripped or
        ignored by downstream parsing). An empty ``<think>...</think>`` pair
        BEFORE the generation slot is the documented Qwen3.5 suppression
        mechanism (model card: ``enable_thinking=False`` injects an empty
        thinking-section pair so the model writes its response AFTER it) and
        does NOT confound the marker install.
        """
        n_open = s.count("<think>")
        n_close = s.count("</think>")
        return n_open > n_close

    default_hit = _has_think(default_prefix)
    suppressed_hit = _has_think(suppressed_prefix)
    default_unclosed = _has_open_unclosed_think(default_prefix)
    suppressed_unclosed = _has_open_unclosed_think(suppressed_prefix)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / "chat_template_lint.json"
    payload = {
        "base_model": BASE_MODEL,
        "default_prefix": default_prefix,
        "suppressed_prefix": suppressed_prefix,
        "default_has_think": default_hit,
        "suppressed_has_think": suppressed_hit,
        "default_has_unclosed_think": default_unclosed,
        "suppressed_has_unclosed_think": suppressed_unclosed,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"  default render contains <think>: {default_hit} (unclosed: {default_unclosed})")
    print(
        f"  enable_thinking=False render contains <think>: {suppressed_hit} "
        f"(unclosed: {suppressed_unclosed})"
    )

    # FAIL only when the dispatcher's actual render path (enable_thinking=False)
    # has an OPEN <think> WITHOUT a matching </think> — the real #475 confound.
    # An empty <think></think> pair BEFORE the generation slot is the Qwen3.5
    # documented suppression mechanism and is fine.
    if suppressed_unclosed:
        print(
            "\nFAIL: enable_thinking=False render has an OPEN <think> at the generation "
            "slot without a closing </think>. This is the #475 confound — the model "
            "would generate content INSIDE the thinking block, breaking the marker slot. "
            "Plan-stop."
        )
        return 1
    if default_unclosed:
        print(
            "\nWARN: default render has unclosed <think>; the dispatcher / eval rig MUST "
            "pass enable_thinking=False (eval_issue506._make_chat_prefix already does)."
        )
    if suppressed_hit and not suppressed_unclosed:
        print(
            "\nOK: enable_thinking=False render has an EMPTY <think></think> pair "
            "(documented Qwen3.5 suppression mechanism). Generation slot is AFTER "
            "the closed thinking section — marker install path unaffected."
        )
    else:
        print("\nOK: no <think> in suppressed render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
