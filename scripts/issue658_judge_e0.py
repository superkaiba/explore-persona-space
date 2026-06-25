#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, →, θ, ×) in scientific docstrings + log messages.
"""Issue #658 J1 (off-pod CPU/API): judge E0(C,B) base expression.

Reads the per-(context, column) generation JSONs the GPU phase wrote under
``eval_results/issue_658/e0_gen/`` and produces the E0(C,B) measurement table
``eval_results/issue_658/E0_expression.json``:

- **PRIMARY DV — judged-rate**: judge each completion with
  ``claude-sonnet-4-5-20250929`` (the standing rule; the testbed legacy
  gpt4o/haiku pins are DEPRECATED) via the Anthropic Batch API for large sets,
  per the column's binary rubric (``issue658_common.E0_COLUMNS``). Rate =
  judge-positive fraction over the (n_samples × n_probes) completions.
- **SECONDARY DV — log-P**: length-normalized base-model log P of the
  judge-POSITIVE on-policy completions (the ``logp_norm`` the GPU phase stored
  per completion). The dual-DV empty-set guard (round-1 concern #3): a column
  with ZERO judged-positive completions has an empty log-P set → emit
  ``low_dynamic_range: true`` + skip the secondary, never crash.
- **marker** column: NO judge — read the 4-float ``marker_slot`` records
  (logp / z_marker / z_eos / logZ); E0(C, marker) = the mean on-policy marker
  log-prob at the end-of-own-response slot.
- **format_style** column: NO judge — the deterministic structural classifier
  (``structural_format_features``); E0 = the is_list_formatted fraction.

GPU-FREE by construction (judge = API, structural = code, marker = read the
stored 4-float records). Honors col.temperature / col.n_samples implicitly (it
judges however many samples the GPU phase produced per column).

Usage::

    uv run python scripts/issue658_judge_e0.py \\
        --e0-dir eval_results/issue_658/e0_gen \\
        --out eval_results/issue_658/E0_expression.json

    # smoke (reads the smoke gen dir, judges with the same rubric):
    uv run python scripts/issue658_judge_e0.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import (  # noqa: E402
    E0_COLUMNS,
    EVAL_RESULTS_DIR,
    JUDGE_MODEL,
    _verdict_truthy,
    dump_json,
    load_json,
)

from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    structural_format_features,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue658_judge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def judge_batch(prompts: list[str], model: str) -> list[dict]:
    """Judge a list of filled rubric prompts with Sonnet 4.5 (threaded API).

    Returns one verdict dict per prompt. Transport/parse failures become a
    tracked ``{"_judge_error": ...}`` (never a silent default). 529 Overloaded
    is transient (the SDK retries internally via max_retries; InternalServerError
    covers it per code-style.md).
    """
    import json
    import re
    from concurrent.futures import ThreadPoolExecutor

    import anthropic

    client = anthropic.Anthropic(max_retries=8)

    def _one(prompt: str) -> dict:
        for attempt in range(2):
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "refusal":
                return {"_judge_refused": "stop_reason=refusal"}
            raw = "\n".join(
                t for t in (getattr(b, "text", None) for b in resp.content) if isinstance(t, str)
            )
            matches = re.findall(r"\{[^{}]*\}", raw, flags=re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[-1])
                except (ValueError, json.JSONDecodeError):
                    pass
            if attempt == 1:
                return {"_judge_error": raw[:200]}
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_one, prompts))


def judge_column(col_id: str, gen: dict, model: str) -> dict:
    """E0(C, B) for one (context, judged column): rate (PRIMARY) + logP (SECONDARY).

    Returns {rate, n_judged, n_positive, logp_pos_mean, low_dynamic_range,
    per_probe, ...}. ``per_probe`` is the PER-PROBE judge-positive fraction over
    that probe's samples — the raw signal the N1 noise floor needs to re-estimate
    E0(C,B) from independent 48-probe redraws (a split-half test-retest of the
    per-behavior E0 estimate, NOT a behavior-agnostic activation-norm proxy —
    the round-2 noise-floor BLOCKER fix). Each entry is {probe, e0, n_judged}.
    """
    col = E0_COLUMNS[col_id]
    # Flatten every (probe, sample) completion with its logp_norm, KEEPING the
    # probe grouping (per_probe[probe] = list of (positive_bool, logp)).
    per_probe_acc: dict[str, list[tuple[bool, float]]] = {}
    flat: list[dict] = []
    for cell in gen["cells"]:
        for comp in cell["completions"]:
            flat.append({"probe": cell["probe"], "text": comp["text"], "logp": comp["logp_norm"]})
    prompts = [col.judge_prompt.format(question=c["probe"], completion=c["text"]) for c in flat]
    verdicts = judge_batch(prompts, model)
    n_judged = 0
    n_positive = 0
    pos_logps: list[float] = []
    for c, v in zip(flat, verdicts, strict=True):
        if "_judge_error" in v or "_judge_refused" in v:
            continue
        n_judged += 1
        pos = _verdict_truthy(v, col.e0_verdict_key, col_id)
        per_probe_acc.setdefault(c["probe"], []).append((pos, c["logp"]))
        if pos:
            n_positive += 1
            pos_logps.append(c["logp"])
    rate = (n_positive / n_judged) if n_judged else None
    # Per-probe E0 contribution: the judge-positive fraction over THAT probe's
    # judged samples (the unit the noise-floor redraws sample from).
    per_probe = [
        {
            "probe": p,
            "e0": (sum(1 for pos, _ in rows if pos) / len(rows)) if rows else None,
            "n_judged": len(rows),
        }
        for p, rows in per_probe_acc.items()
    ]
    # Dual-DV empty-set guard (round-1 concern #3): zero judged-positives -> the
    # log-P companion is the empty set -> low_dynamic_range, skip the secondary.
    low_dyn = (not pos_logps) or (rate in (0.0, 1.0))
    return {
        "column_id": col_id,
        "dv": "judged_rate",
        "rate": rate,  # PRIMARY
        "n_judged": n_judged,
        "n_positive": n_positive,
        "n_total": len(flat),
        "logp_pos_mean": (sum(pos_logps) / len(pos_logps)) if pos_logps else None,  # SECONDARY
        "low_dynamic_range": low_dyn,
        "per_probe": per_probe,  # N1 redraw unit (round-2 noise-floor fix)
    }


def score_format(gen: dict) -> dict:
    """E0 for format_style: deterministic structural classifier (no judge)."""
    flags = []
    logps = []
    per_probe: list[dict] = []
    for cell in gen["cells"]:
        cell_flags = []
        for comp in cell["completions"]:
            f = structural_format_features(comp["text"])["is_list_formatted"]
            flags.append(f)
            cell_flags.append(f)
            logps.append(comp["logp_norm"])
        per_probe.append(
            {
                "probe": cell["probe"],
                "e0": (sum(1 for f in cell_flags if f) / len(cell_flags)) if cell_flags else None,
                "n_judged": len(cell_flags),
            }
        )
    rate = (sum(1 for f in flags if f) / len(flags)) if flags else None
    pos_logps = [lp for f, lp in zip(flags, logps, strict=True) if f]
    return {
        "column_id": "format_style",
        "dv": "structural",
        "rate": rate,
        "n_total": len(flags),
        "logp_pos_mean": (sum(pos_logps) / len(pos_logps)) if pos_logps else None,
        "low_dynamic_range": (not pos_logps) or (rate in (0.0, 1.0)),
        "per_probe": per_probe,  # N1 redraw unit (round-2 noise-floor fix)
    }


def score_marker(gen: dict) -> dict:
    """E0(C, marker): mean on-policy marker log-prob at end-of-own-response slot.

    ``per_probe`` carries the per-probe marker log-prob — the redraw unit the
    N1 noise floor samples for the marker column (the marker E0 target is a
    continuous logp, not a judged rate, so its per-probe contribution is the
    raw slot logp; round-2 noise-floor fix).
    """
    logps = [r["logp"] for r in gen["marker_slot"]]
    z_margins = [r["z_marker"] - r["z_eos"] for r in gen["marker_slot"]]
    emits = [r.get("argmax_id") == 83399 for r in gen["marker_slot"]]
    per_probe = [{"probe": r["probe"], "e0": r["logp"], "n_judged": 1} for r in gen["marker_slot"]]
    return {
        "column_id": "marker",
        "dv": "marker_slot_stats",
        "logp_mean": (sum(logps) / len(logps)) if logps else None,  # PRIMARY (marker DV)
        "eos_margin_mean": (sum(z_margins) / len(z_margins)) if z_margins else None,  # SECONDARY
        "emission_rate": (sum(1 for e in emits if e) / len(emits)) if emits else None,
        "n_total": len(logps),
        "per_probe": per_probe,  # N1 redraw unit (round-2 noise-floor fix)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #658 J1: judge E0(C,B).")
    parser.add_argument("--e0-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    e0_dir = args.e0_dir or (EVAL_RESULTS_DIR / ("e0_gen_smoke" if args.smoke else "e0_gen"))
    out = args.out or (
        EVAL_RESULTS_DIR / ("E0_expression_smoke.json" if args.smoke else "E0_expression.json")
    )
    gen_files = sorted(e0_dir.glob("*__*.json"))
    if not gen_files:
        raise RuntimeError(f"no E0 generation files under {e0_dir} (run the extractor first)")

    results: dict = {}  # {context_id: {column_id: {...}}}
    for gf in gen_files:
        gen = load_json(gf)
        ctx = gen["context_id"]
        col_id = gen["column_id"]
        results.setdefault(ctx, {})
        if col_id == "marker":
            results[ctx][col_id] = score_marker(gen)
        elif col_id == "format_style":
            results[ctx][col_id] = score_format(gen)
        else:
            results[ctx][col_id] = judge_column(col_id, gen, args.model)
        logger.info("E0[%s][%s]: %s", ctx, col_id, results[ctx][col_id].get("rate"))

    payload = {
        "judge_model": args.model,
        "e0": results,
        "columns": list(E0_COLUMNS.keys()),
        "dual_dv": {
            "primary": "judged_rate / marker logp",
            "secondary": "length-normalized base-model log P of judged-positive completions",
            "empty_set_guard": "low_dynamic_range=true when no judged-positive completions exist",
        },
        "metadata": reproducibility_metadata({"script": "issue658_judge_e0"}),
    }
    dump_json(payload, out)
    n_low = sum(1 for c in results.values() for v in c.values() if v.get("low_dynamic_range"))
    logger.info(
        "Wrote %s (%d contexts; %d low-dynamic-range cells flagged)", out, len(results), n_low
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
