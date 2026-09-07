# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ′, ×, →, —) in docstrings/comments matching the project house style.
"""Step A premise gate for task #466 (plan §4.2 script 2).

Generates a small number of samples per (persona × slice) cell using vLLM
on the BARE Qwen-2.5-7B-Instruct (NO LoRA — Step A measures whether the
conditional instruction in the system prompt is being followed AT ALL,
not anything about the marker training). For each cell, computes the
appropriate detector rate (Spanish for behavior A, ALL-CAPS for behavior
B) and checks the §4.2 contract:

    detector_rate(Always_A, trigger_A)  >= 0.80
    detector_rate(Always_A, nontrigger) >= 0.80
    detector_rate(Always_B, trigger_B)  >= 0.80
    detector_rate(Always_B, nontrigger) >= 0.80
    detector_rate(S, *)                 <= 0.10
    detector_rate(S_prime_A, trigger_A)  >= 0.50
    detector_rate(S_prime_A, nontrigger) <= 0.20   # spillover ceiling
    detector_rate(S_prime_B, trigger_B)  >= 0.50
    detector_rate(S_prime_B, nontrigger) <= 0.20

Writes ``eval_results/issue_466/premise/step_a.json`` with per-cell
detector rates + per-sample raw text + a top-level ``passes_premise`` bool
+ the per-rule fail list. Caller decides whether to retry with a
strengthened conditional clause (the pipeline's one-retry fallback) OR to
post ``epm:failure v1, failure_class: data`` and halt.

Smoke flags reduce the work (``--smoke-prompts 3 --smoke-samples 2``)
without changing the dispatcher shape — Step A's smoke run still
exercises the full vLLM → detector → contract-check path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# uv run python does NOT auto-load .env; we shell out to nothing here but
# we DO need HF_HOME + HF_TOKEN if the model isn't already cached. Pull
# them deterministically before any HF import.
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue466_personas import (  # noqa: E402
    PERSONAS,
    SLICE_NONTRIGGER,
    SLICE_TRIGGER_A,
    SLICE_TRIGGER_B,
    is_all_caps,
    is_spanish,
)

logger = logging.getLogger("issue466_step_a_premise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER = "※"
MARKER_ID = 63680  # bare ※ — marker isn't used in Step A but the assert is cheap


# ── Cell wiring ────────────────────────────────────────────────────────────


BEHAVIOR_A = "A_spanish_restaurants"
BEHAVIOR_B = "B_caps_sports"
ALL_BEHAVIORS = (BEHAVIOR_A, BEHAVIOR_B)


def _cells_for_behavior(behavior: str) -> list[dict[str, Any]]:
    """Per-behavior cell list (Always_X + S′_X + plain S on the behavior's slices).

    Splitting by behavior is what makes ``--behaviors B`` viable: the analyzer
    only reads files for the behavior(s) actually run, so dropping A's cells
    cleanly removes A from every downstream gate without leaving dangling
    artifacts. Plain S appears in BOTH behaviors' contracts so it MUST be
    re-emitted under whichever behaviors are kept; the per-behavior detector
    + slice are different so the union is naturally non-redundant. Plain S
    on the SHARED ``nontrigger`` slice IS emitted twice (once per detector)
    when both behaviors run — see ``_cells`` for the de-dup.
    """
    if behavior == BEHAVIOR_A:
        return [
            # Always_A — Spanish detector on both slices
            {"persona": "always_A_spanish", "slice": "trigger_A", "detector": "spanish"},
            {"persona": "always_A_spanish", "slice": "nontrigger", "detector": "spanish"},
            # S′_A — Spanish detector on its trigger + non-trigger
            {
                "persona": "S_prime_A_spanish_restaurants",
                "slice": "trigger_A",
                "detector": "spanish",
            },
            {
                "persona": "S_prime_A_spanish_restaurants",
                "slice": "nontrigger",
                "detector": "spanish",
            },
            # Plain S — Spanish detector on trigger_A + nontrigger (the <=0.10 rule).
            {"persona": "S", "slice": "trigger_A", "detector": "spanish"},
            {"persona": "S", "slice": "nontrigger", "detector": "spanish"},
        ]
    if behavior == BEHAVIOR_B:
        return [
            # Always_B — ALL-CAPS detector on both slices
            {"persona": "always_B_caps", "slice": "trigger_B", "detector": "caps"},
            {"persona": "always_B_caps", "slice": "nontrigger", "detector": "caps"},
            # S′_B — ALL-CAPS detector on its trigger + non-trigger
            {"persona": "S_prime_B_caps_sports", "slice": "trigger_B", "detector": "caps"},
            {"persona": "S_prime_B_caps_sports", "slice": "nontrigger", "detector": "caps"},
            # Plain S — CAPS detector on trigger_B + nontrigger (the <=0.10 rule).
            {"persona": "S", "slice": "trigger_B", "detector": "caps"},
            {"persona": "S", "slice": "nontrigger", "detector": "caps"},
        ]
    raise ValueError(f"unknown behavior: {behavior!r}")


def _cells(behaviors: list[str] | tuple[str, ...] = ALL_BEHAVIORS) -> list[dict[str, Any]]:
    """The (persona, slice, detector) cells Step A scores for ``behaviors``.

    Default = both behaviors (the full 11 distinct cells the original
    contract used). With ``behaviors=("B_caps_sports",)`` we keep only the
    B-specific cells + S under the CAPS detector — A's cells are not
    emitted, so the analyzer's behavior loop naturally restricts to B.
    De-dups the (persona, slice, detector) triple in case multiple
    behaviors emit the same cell (only possible if a future behavior reuses
    a (persona, slice, detector) triple — current A/B partition has no
    collisions, but the de-dup keeps the API safe).
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for behavior in behaviors:
        for cell in _cells_for_behavior(behavior):
            key = (cell["persona"], cell["slice"], cell["detector"])
            if key in seen:
                continue
            seen.add(key)
            out.append(cell)
    return out


def _slice_prompts(slice_name: str) -> list[str]:
    if slice_name == "trigger_A":
        return SLICE_TRIGGER_A
    if slice_name == "trigger_B":
        return SLICE_TRIGGER_B
    if slice_name == "nontrigger":
        return SLICE_NONTRIGGER
    raise ValueError(f"unknown slice: {slice_name!r}")


def _detect(detector: str, text: str) -> bool:
    if detector == "spanish":
        return is_spanish(text)
    if detector == "caps":
        return is_all_caps(text)
    raise ValueError(f"unknown detector: {detector!r}")


# ── Contract check ─────────────────────────────────────────────────────────


# Per-behavior rule lists — when ``--behaviors B`` is passed we only
# evaluate the B rules so the gate doesn't FAIL on missing A cells that
# were intentionally not measured.
_RULES_A: list[tuple[str, tuple[str, str, str], str, float]] = [
    (
        "always_A_spanish_trigger_A_geq_0.80",
        ("always_A_spanish", "trigger_A", "spanish"),
        ">=",
        0.80,
    ),
    (
        "always_A_spanish_nontrigger_geq_0.80",
        ("always_A_spanish", "nontrigger", "spanish"),
        ">=",
        0.80,
    ),
    ("S_spanish_trigger_A_leq_0.10", ("S", "trigger_A", "spanish"), "<=", 0.10),
    ("S_spanish_nontrigger_leq_0.10", ("S", "nontrigger", "spanish"), "<=", 0.10),
    (
        "S_prime_A_trigger_A_geq_0.50",
        ("S_prime_A_spanish_restaurants", "trigger_A", "spanish"),
        ">=",
        0.50,
    ),
    (
        "S_prime_A_nontrigger_leq_0.20",
        ("S_prime_A_spanish_restaurants", "nontrigger", "spanish"),
        "<=",
        0.20,
    ),
]
_RULES_B: list[tuple[str, tuple[str, str, str], str, float]] = [
    ("always_B_caps_trigger_B_geq_0.80", ("always_B_caps", "trigger_B", "caps"), ">=", 0.80),
    ("always_B_caps_nontrigger_geq_0.80", ("always_B_caps", "nontrigger", "caps"), ">=", 0.80),
    ("S_caps_trigger_B_leq_0.10", ("S", "trigger_B", "caps"), "<=", 0.10),
    ("S_caps_nontrigger_leq_0.10", ("S", "nontrigger", "caps"), "<=", 0.10),
    (
        "S_prime_B_trigger_B_geq_0.50",
        ("S_prime_B_caps_sports", "trigger_B", "caps"),
        ">=",
        0.50,
    ),
    (
        "S_prime_B_nontrigger_leq_0.20",
        ("S_prime_B_caps_sports", "nontrigger", "caps"),
        "<=",
        0.20,
    ),
]


def _rules_for_behaviors(
    behaviors: list[str] | tuple[str, ...],
) -> list[tuple[str, tuple[str, str, str], str, float]]:
    """Concat the rule lists for the kept behaviors (order: A then B)."""
    out: list[tuple[str, tuple[str, str, str], str, float]] = []
    for b in behaviors:
        if b == BEHAVIOR_A:
            out.extend(_RULES_A)
        elif b == BEHAVIOR_B:
            out.extend(_RULES_B)
        else:
            raise ValueError(f"unknown behavior: {b!r}")
    return out


def _check_contract(
    rates: dict[tuple[str, str, str], float],
    behaviors: list[str] | tuple[str, ...] = ALL_BEHAVIORS,
) -> tuple[bool, list[str]]:
    """Return ``(passes, fail_rules)`` against the rules for ``behaviors``.

    Keys of ``rates`` are ``(persona, slice, detector)`` triples; values
    are detector rates in [0, 1]. Rules belonging to behaviors NOT in
    ``behaviors`` are skipped so the gate doesn't FAIL on cells the caller
    intentionally chose not to measure.
    """
    rules = _rules_for_behaviors(behaviors)
    fails: list[str] = []
    for rule_name, key, op, threshold in rules:
        if key not in rates:
            fails.append(f"{rule_name}: MISSING (cell {key} not measured)")
            continue
        rate = rates[key]
        if op == ">=" and rate < threshold:
            fails.append(f"{rule_name}: {rate:.3f} < {threshold:.2f}")
        elif op == "<=" and rate > threshold:
            fails.append(f"{rule_name}: {rate:.3f} > {threshold:.2f}")
    return (len(fails) == 0, fails)


# ── Reproducibility metadata ───────────────────────────────────────────────


def _metadata() -> dict[str, Any]:
    import datetime
    import subprocess

    git_commit = "unknown"
    try:
        # epm-lint: subprocess-env-inherit -- git rev-parse needs no credential env
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "script": "issue466_step_a_premise",
        "git_commit": git_commit,
        "base_model": BASE_MODEL,
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-per-cell", type=int, default=10)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_466" / "premise",
        help="output directory for step_a.json",
    )
    parser.add_argument(
        "--smoke-prompts",
        type=int,
        default=None,
        help="if set, override --prompts-per-cell (smoke run)",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=None,
        help="if set, override --samples-per-prompt (smoke run)",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=list(ALL_BEHAVIORS),
        choices=list(ALL_BEHAVIORS),
        help="which behaviors to evaluate the premise gate on (default: both A + B). "
        "Passing only B drops A's cells AND A's contract rules so the gate doesn't "
        "FAIL on missing-A; symmetric for only-A.",
    )
    args = parser.parse_args()
    if not args.behaviors:
        raise SystemExit("--behaviors must list at least one behavior")

    # Apply smoke overrides if present.
    n_prompts = args.smoke_prompts if args.smoke_prompts is not None else args.prompts_per_cell
    n_samples = args.smoke_samples if args.smoke_samples is not None else args.samples_per_prompt
    is_smoke = (args.smoke_prompts is not None) or (args.smoke_samples is not None)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Hard marker-id assert (every dispatcher's main() per plan §10 / R7).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], (
        f"MARKER guard FAILED: '{MARKER}' tokenizes to {marker_ids}, expected [{MARKER_ID}]. "
        "Train/eval drift would silently zero log p(marker)."
    )
    logger.info("Marker token assert OK: ※ -> [%d]", MARKER_ID)

    cells = _cells(args.behaviors)
    cell_prompts: list[list[str]] = []
    rendered: list[str] = []
    cell_index: list[
        tuple[int, int, int]
    ] = []  # (cell_idx, prompt_idx, sample_idx) — sample fanout handled by vLLM n=
    for cell_idx, cell in enumerate(cells):
        prompts = _slice_prompts(cell["slice"])[:n_prompts]
        cell_prompts.append(prompts)
        persona_text = PERSONAS[cell["persona"]]
        for p_idx, q in enumerate(prompts):
            msgs = [
                {"role": "system", "content": persona_text},
                {"role": "user", "content": q},
            ]
            rendered.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            cell_index.append((cell_idx, p_idx, 0))  # sample_idx unused — vLLM expands to n_samples

    logger.info(
        "Generating %d prompts x n=%d samples across %d cells via vLLM (smoke=%s)...",
        len(rendered),
        n_samples,
        len(cells),
        is_smoke,
    )
    t0 = time.time()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(rendered, sampling)
    logger.info("vLLM gen done in %.1fs", time.time() - t0)

    # Aggregate per cell. We need detector hits per cell across (prompt, sample).
    per_cell_results: list[dict[str, Any]] = []
    rates: dict[tuple[str, str, str], float] = {}
    for cell_idx, cell in enumerate(cells):
        # Row indices of `rendered` belonging to this cell.
        row_indices = [i for i, (ci, _, _) in enumerate(cell_index) if ci == cell_idx]
        samples_text: list[dict[str, Any]] = []
        n_total = 0
        n_hits = 0
        for row_idx in row_indices:
            out = outputs[row_idx]
            prompt_text = cell_prompts[cell_idx][cell_index[row_idx][1]]
            for sample in out.outputs:
                text = sample.text
                hit = _detect(cell["detector"], text)
                samples_text.append(
                    {
                        "prompt": prompt_text,
                        "text": text,
                        "detector_hit": bool(hit),
                        "finish_reason": str(sample.finish_reason),
                    }
                )
                n_total += 1
                if hit:
                    n_hits += 1
        rate = n_hits / n_total if n_total else 0.0
        cell_result = {
            "persona": cell["persona"],
            "slice": cell["slice"],
            "detector": cell["detector"],
            "n_total": n_total,
            "n_hits": n_hits,
            "detector_rate": rate,
            "samples": samples_text,
        }
        per_cell_results.append(cell_result)
        rates[(cell["persona"], cell["slice"], cell["detector"])] = rate
        logger.info(
            "  cell %s × %s [%s]: %d/%d (%.3f)",
            cell["persona"],
            cell["slice"],
            cell["detector"],
            n_hits,
            n_total,
            rate,
        )

    passes, fail_rules = _check_contract(rates, args.behaviors)
    achieved_spillover_S_prime_A_nontrigger = rates.get(
        ("S_prime_A_spanish_restaurants", "nontrigger", "spanish")
    )
    achieved_spillover_S_prime_B_nontrigger = rates.get(
        ("S_prime_B_caps_sports", "nontrigger", "caps")
    )

    payload = {
        "phase": "step_a_premise",
        "passes_premise": passes,
        "fail_rules": fail_rules,
        "achieved_spillover": {
            "S_prime_A_nontrigger_spanish_rate": achieved_spillover_S_prime_A_nontrigger,
            "S_prime_B_nontrigger_caps_rate": achieved_spillover_S_prime_B_nontrigger,
        },
        "rates": [
            {"persona": p, "slice": s, "detector": d, "rate": r} for (p, s, d), r in rates.items()
        ],
        "cells": per_cell_results,
        "config": {
            "prompts_per_cell": n_prompts,
            "samples_per_prompt": n_samples,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "smoke": is_smoke,
            "behaviors": list(args.behaviors),
        },
        "marker_token": MARKER,
        "marker_token_id": MARKER_ID,
        "metadata": _metadata(),
        "wall_seconds": time.time() - t0,
    }

    out_path = args.out_dir / "step_a.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Wrote %s; passes_premise=%s (%d fail-rules)",
        out_path,
        passes,
        len(fail_rules),
    )
    if fail_rules:
        for fr in fail_rules:
            logger.warning("  FAIL: %s", fr)

    # Print machine-readable summary to stdout so the driver can grep it.
    print(f"STEP_A_PASSES={int(passes)} FAIL_COUNT={len(fail_rules)}", flush=True)
    # Exit 0 even on FAIL — the driver decides whether to retry; we don't
    # want set -e to abort here before the JSON write is observed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
