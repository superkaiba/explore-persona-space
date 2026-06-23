#!/usr/bin/env python
"""Measure the JS/KL outdist prompt-token lengths for the #545 metric-race.

Strategy-D pre-check (#545 round 37): before lowering ``JS_MAX_SEQ_LEN``
(the vLLM ``max_model_len`` for the lazily-loaded engine inside
``extract_clouds_and_outdist_gpu``), measure the ACTUAL maximum prompt
length the scoring path constructs. ``JS_MAX_SEQ_LEN`` bounds vLLM's
KV-cache AND the longest sequence vLLM accepts (prompt + generated), so a
value below the worst-case ``prompt + JS_MAX_NEW_TOKENS`` either crashes
vLLM on the oversize request or drops it — silently corrupting the scoring
of the affected cells.

The scoring prompt is ``apply_chat_template([*ctx, {user: probe}])`` where
``ctx`` is the row's behavior context (``nl`` system turn OR ``demos``
K=8 few-shot prefix — the long one) and ``probe`` is the column's battery
probe (the ``deception`` column embeds full code/negotiation transcripts —
the other long one). This script reproduces the exact
``_score_outdist_pair`` / clouds enumeration on CPU (tokenizer only, no
GPU) and reports the length distribution + the per-threshold truncation
counts that decide whether any ``JS_MAX_SEQ_LEN`` reduction is safe.

CPU-only; run on the VM. Reproducibility metadata emitted in the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# repo-root src on path (this script lives in scripts/)
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.experiments.behavior_testbed_545 import (  # noqa: E402
    BASE_MODEL,
    reproducibility_metadata,
)
from explore_persona_space.experiments.behavior_testbed_545.columns import (  # noqa: E402
    column_applies,
)
from explore_persona_space.experiments.behavior_testbed_545.predictors_zoo import (  # noqa: E402
    COLUMNS,
    FLAVORS,
    JS_MAX_NEW_TOKENS,
    JS_MAX_SEQ_LEN,
    JS_N_PROBES,
    ROWS,
    _behavior_context_messages,
    _column_probe_texts,
)

_THRESHOLDS = (1024, 2048, 3072, 4096, 5120, 6144, 8192)


def _measure(tokenizer, n_probes: int, scoring_only: bool) -> dict:
    def prompt_len(ctx: list[dict], q: str) -> int:
        msgs = [*ctx, {"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text, add_special_tokens=False))

    if scoring_only:
        col_ids = [c for c, col in COLUMNS.items() if col.scoring_eligible]
    else:
        col_ids = list(COLUMNS.keys())

    probe_cache: dict[str, list[str]] = {}
    probe_errors: dict[str, str] = {}
    for col_id in col_ids:
        try:
            probe_cache[col_id] = _column_probe_texts(col_id, cap=n_probes)
        except Exception as exc:
            probe_cache[col_id] = []
            probe_errors[col_id] = f"{type(exc).__name__}: {exc}"

    lens: list[int] = []
    max_prompt = 0
    max_where: tuple | None = None
    for flavor in FLAVORS:
        for row_id in ROWS:
            ctx = _behavior_context_messages(row_id, flavor)
            if not ctx:
                continue
            for col_id in col_ids:
                if not column_applies(COLUMNS[col_id], ROWS[row_id]):
                    continue
                for q in probe_cache[col_id]:
                    length = prompt_len(ctx, q)
                    lens.append(length)
                    if length > max_prompt:
                        max_prompt = length
                        max_where = (flavor, row_id, col_id)

    if not lens:
        raise RuntimeError("zero prompts enumerated — column/probe wiring broken")

    arr = np.array(lens)
    pcts = {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99, 99.9, 100)}
    thr = {}
    for t in _THRESHOLDS:
        over = int((arr > t).sum())
        over_pm = int((arr + JS_MAX_NEW_TOKENS > t).sum())
        thr[str(t)] = {
            "prompts_over": over,
            "prompts_over_frac": round(over / len(arr), 4),
            "prompt_plus_maxnew_over": over_pm,
            "prompt_plus_maxnew_over_frac": round(over_pm / len(arr), 4),
        }

    return {
        "scoring_only": scoring_only,
        "n_columns": len(col_ids),
        "n_prompts": len(arr),
        "max_prompt_tokens": int(max_prompt),
        "max_prompt_where": {"flavor": max_where[0], "row": max_where[1], "col": max_where[2]}
        if max_where
        else None,
        "worst_case_prompt_plus_maxnew": int(max_prompt + JS_MAX_NEW_TOKENS),
        "js_max_new_tokens": JS_MAX_NEW_TOKENS,
        "current_js_max_seq_len": JS_MAX_SEQ_LEN,
        "percentiles": pcts,
        "thresholds": thr,
        "probe_errors": probe_errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure #545 JS outdist prompt lengths (CPU).")
    ap.add_argument("--n-probes", type=int, default=JS_N_PROBES)
    ap.add_argument(
        "--all-columns",
        action="store_true",
        help="measure every column, not just scoring_eligible ones",
    )
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    result = _measure(tokenizer, n_probes=args.n_probes, scoring_only=not args.all_columns)
    result["base_model"] = BASE_MODEL
    result["metadata"] = reproducibility_metadata()

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
