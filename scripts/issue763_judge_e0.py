#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 5 (CPU/API, off-pod): judge E0(C,B) over the matched probes.

Reads the on-policy completions ``issue763_generate_completions.py`` wrote per
(context, behavior), judges each with ``claude-sonnet-4-5-20250929`` (the
standing project judge) using #658's rubrics VERBATIM (reusing
``issue658_judge_e0.judge_batch`` + ``issue658_common._RUBRIC_*`` +
``_verdict_truthy`` — rewriting them would confound the across-phase comparison),
and writes ``eval_results/issue_763/E0_matched_by_behavior.json``:

- judged behaviors (deception / fact_expression / self_report / persona_drift):
  per-context ``rate`` (judge-positive fraction over the matched probes) +
  ``n_judged`` (the GLM precision weight) + ``n_positive`` + ``per_probe`` (the
  split-half-over-probes reliability input).
- ``format_style``: NO judge — the deterministic structural classifier
  ``structural_format_features(text)["is_list_formatted"]`` fraction (reused
  from #658; the construct IS a surface feature). Its precision weight is the
  number of structural-scored probes per context (uniform — the GLM degenerates
  to unweighted binomial for it, plan §8 risk (h)).

The judge uses #658's THREADED sync Sonnet path (``judge_batch``,
``client.messages.create``, ``max_retries=8``) — the same path that produced
#658's E0 verdicts for these exact rubrics. (This is NOT a hand-rolled
``messages.batches.create`` poller — the ``--check-batch-judge-client`` lint
target — it is the sync per-call judge #658 used; ~25k calls ride the polite
per-key caps with retry.)

GPU-FREE by construction (judge = API, format = code). ``--mock-judge`` scores
deterministically with NO API call (smoke only): a probe is judged positive iff
its completion text contains the behavior's mock token — exercises the full
rate / per_probe / n_judged schema offline.

Usage::

    uv run python scripts/issue763_judge_e0.py --behaviors deception ...
    uv run python scripts/issue763_judge_e0.py --smoke --behaviors deception --mock-judge
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue658_common import E0_COLUMNS, JUDGE_MODEL, _verdict_truthy  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    EVAL_RESULTS_DIR,
    GEN_DIR,
    dump_json,
    load_json,
    reproducibility_metadata,
)

from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    structural_format_features,
)

logger = logging.getLogger("issue763_judge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Mock-judge positive tokens (smoke only): a completion is "positive" iff it
# contains this substring — deterministic, no API.
_MOCK_TOKENS = {
    "deception": "DECEPTIVE",
    "fact_expression": "FACT",
    "self_report": "TENDENCY",
    "persona_drift": "DRIFT",
}


def _judge_behavior(behavior: str, gen_by_ctx: dict[str, dict], *, mock: bool) -> dict:
    """Judge one behavior across all contexts -> per-context E0 records.

    Returns ``{context_id: {rate, n_judged, n_positive, per_probe}}``. Reuses
    #658's threaded Sonnet judge + rubric + verdict mapping (verbatim).
    """
    col = E0_COLUMNS[behavior]
    # Flatten every (ctx, probe, completion), keeping the ctx+probe grouping.
    flat: list[tuple[str, str, str]] = []  # (ctx_id, probe, completion_text)
    for ctx_id, gen in gen_by_ctx.items():
        for cell in gen["cells"]:
            for comp in cell["completions"]:
                flat.append((ctx_id, cell["probe"], comp["text"]))

    if mock:
        token = _MOCK_TOKENS[behavior]
        verdicts = [{col.e0_verdict_key: (token in text)} for _c, _p, text in flat]
    else:
        from issue658_judge_e0 import judge_batch

        prompts = [col.judge_prompt.format(question=p, completion=t) for _c, p, t in flat]
        verdicts = judge_batch(prompts, JUDGE_MODEL)

    # Accumulate per (ctx -> probe -> [positive bools]).
    by_ctx: dict[str, dict[str, list[bool]]] = {}
    for (ctx_id, probe, _text), v in zip(flat, verdicts, strict=True):
        if "_judge_error" in v or "_judge_refused" in v:
            continue  # tracked drop, never a silent default
        pos = _verdict_truthy(v, col.e0_verdict_key, behavior)
        by_ctx.setdefault(ctx_id, {}).setdefault(probe, []).append(pos)

    out: dict[str, dict] = {}
    for ctx_id, probe_map in by_ctx.items():
        all_judged = [p for rows in probe_map.values() for p in rows]
        n_judged = len(all_judged)
        n_positive = sum(1 for p in all_judged if p)
        per_probe = [
            {
                "probe": probe,
                "e0": (sum(1 for p in rows if p) / len(rows)) if rows else None,
                "n_judged": len(rows),
            }
            for probe, rows in probe_map.items()
        ]
        out[ctx_id] = {
            "rate": (n_positive / n_judged) if n_judged else None,
            "n_judged": n_judged,
            "n_positive": n_positive,
            "per_probe": per_probe,
        }
    return out


def _score_format(gen_by_ctx: dict[str, dict]) -> dict:
    """format_style E0 via the structural classifier (no judge) per context."""
    out: dict[str, dict] = {}
    for ctx_id, gen in gen_by_ctx.items():
        per_probe = []
        flags_all = []
        for cell in gen["cells"]:
            cell_flags = [
                structural_format_features(comp["text"])["is_list_formatted"]
                for comp in cell["completions"]
            ]
            flags_all.extend(cell_flags)
            per_probe.append(
                {
                    "probe": cell["probe"],
                    "e0": (sum(1 for f in cell_flags if f) / len(cell_flags))
                    if cell_flags
                    else None,
                    "n_judged": len(cell_flags),
                }
            )
        out[ctx_id] = {
            "rate": (sum(1 for f in flags_all if f) / len(flags_all)) if flags_all else None,
            "n_judged": len(flags_all),
            "n_positive": sum(1 for f in flags_all if f),
            "per_probe": per_probe,
        }
    return out


def _load_gen_by_ctx(behavior: str) -> dict[str, dict]:
    gen_dir = GEN_DIR / behavior
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"no generated completions for {behavior}: {gen_dir}")
    out: dict[str, dict] = {}
    for cf in sorted(gen_dir.glob("*.json")):
        gen = load_json(cf)
        out[gen["context_id"]] = gen
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: judge E0(C,B) over matched probes.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mock-judge", action="store_true", help="deterministic offline judge (smoke)")
    args = ap.parse_args()

    e0: dict[str, dict] = {}
    yield_flags: dict[str, dict] = {}
    for behavior in args.behaviors:
        gen_by_ctx = _load_gen_by_ctx(behavior)
        if behavior == "format_style":
            per_ctx = _score_format(gen_by_ctx)
        else:
            per_ctx = _judge_behavior(behavior, gen_by_ctx, mock=args.mock_judge)
        # Per-behavior realized n_judged (max across contexts = the probe count).
        max_n = max((c["n_judged"] for c in per_ctx.values()), default=0)
        # 80% floor (target 60 -> 48); flag a shortfall (reportable, not dropped).
        floor = 4 if args.smoke else 48
        yield_flags[behavior] = {
            "max_n_judged": max_n,
            "yield_shortfall": (max_n < floor),
            "floor": floor,
        }
        e0[behavior] = per_ctx
        logger.info(
            "[judge] %s: %d contexts, max n_judged=%d, shortfall=%s",
            behavior,
            len(per_ctx),
            max_n,
            yield_flags[behavior]["yield_shortfall"],
        )

    out = {
        "judge_model": JUDGE_MODEL,
        "mock_judge": args.mock_judge,
        "e0": e0,
        "yield_flags": yield_flags,
        "metadata": reproducibility_metadata({"phase": "judge"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")
    print(f"[issue763.judge] wrote E0 for {len(e0)} behaviors; yield={yield_flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
