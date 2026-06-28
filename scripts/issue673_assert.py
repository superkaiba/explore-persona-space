#!/usr/bin/env python3
"""Offline test-verdict reader for the #673 real-GPU memory curves.

Reads the per-regime ``memory_curves_*.json`` written by
``scripts/issue673_gpu_memory_validation.py`` and emits ONE of three verdicts
(plan §6.2):

- ``PASS`` (rc 0): the hook arm is flat under BOTH allocator regimes AND the
  positive control discriminates (under ``expandable_segments:True`` the old arm
  retains materially more reserved memory than the hook arm, OR shows a positive
  monotone trend the hook arm does not).
- ``INCONCLUSIVE`` (rc 0, ``INCONCLUSIVE: ...`` message): the hook arm is flat
  but the positive control did not discriminate at this scale (no false PASS —
  the benchmark at N=50 / one model copy did not exhibit the #545 growth).
- ``REGRESSION`` (rc 1): the hook arm grows under at least one allocator regime
  — a #671 regression. Do NOT land a "PASS".

The reader also asserts BOTH arms ran grad-disabled (``grad_enabled: false`` +
``inference_mode: true`` per arm); a record with ``grad_enabled: true`` invalidates
the positive-control comparison and raises an ``AssertionError`` (the Must-Fix
isolation guarantee — an autograd-graph-inflated gap must never be read as a real
``output_hidden_states`` retention gap).

It also generates the two reserved/allocated line PNGs (one per allocator regime)
under ``figures/issue_673/`` via the paper-plots style.

Usage::

    # two per-regime JSONs (hook + old arms each carry both arms; one JSON per regime):
    uv run python scripts/issue673_assert.py \
      --expandable-json figures/issue_673/memory_curves_expandable_segments_on.json \
      --default-json    figures/issue_673/memory_curves_default_allocator.json

    # verdict only, no figures:
    uv run python scripts/issue673_assert.py --expandable-json <p> --default-json <p> --no-figures
"""

# ruff: noqa: RUF003  # scientific notation (≥, −, ≈) in docstrings/strings

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

# Re-use the canonical constants + flat() logic from the benchmark script so the
# gate strength is defined in exactly ONE place (the test pins these values).
# Repo root must be on sys.path so `scripts.issue673_gpu_memory_validation`
# resolves when this script is invoked as `uv run python scripts/issue673_assert.py`
# (Python only adds the script's own directory, scripts/, to sys.path by default —
# repo root is needed for the `scripts.` package import).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.issue673_gpu_memory_validation import (
    WARMUP,
    ABS_TOL_GiB,
    CTRL_GAP_GiB,
    GiB,
    SLOPE_TOL_GiB_per_iter,
    flat,
)

PASS = "PASS"
INCONCLUSIVE = "INCONCLUSIVE"
REGRESSION = "REGRESSION"


def _assert_grad_disabled(results: dict, regime_tag: str) -> None:
    """Raise if either arm did not run grad-disabled (the Must-Fix isolation guard).

    Both arms MUST carry ``grad_enabled: false`` + ``inference_mode: true``; a
    ``grad_enabled: true`` record means the positive-control gap could be an
    autograd-graph artifact, not the ``output_hidden_states`` tuple keepalive,
    so the comparison is invalid.
    """
    for arm_name in ("hook", "old_ohs_true"):
        arm = results["arms"][arm_name]
        assert arm.get("grad_enabled") is False, (
            f"{regime_tag}/{arm_name}: grad_enabled must be False "
            f"(got {arm.get('grad_enabled')!r}) — an autograd-inflated gap is not a "
            f"valid positive control"
        )
        assert arm.get("inference_mode") is True, (
            f"{regime_tag}/{arm_name}: inference_mode must be True "
            f"(got {arm.get('inference_mode')!r})"
        )


def _assert_allocator_tag(results: dict, expected_tag: str) -> None:
    """Raise if the JSON's allocator_tag does not match the expected regime.

    Guards against a swapped --expandable-json / --default-json invocation or
    a duplicated regime run silently producing a false PASS at the test-verdict
    gate (Codex code-review round 1). The benchmark writes ``allocator_tag``
    (``expandable_segments_on`` | ``default_allocator``) at write time; this is
    the reader-side enforcement.
    """
    got = results.get("allocator_tag")
    assert got == expected_tag, (
        f"allocator regime mismatch: expected {expected_tag!r}, got {got!r}. "
        f"A swapped --expandable-json / --default-json (or a duplicated-regime "
        f"run) would silently produce a false PASS — fail loud instead."
    )


def _max_reserved_gib(reserved_bytes) -> float:
    """Max reserved memory (GiB) over the post-warmup window.

    Mirrors ``flat()``'s short-curve guard in the benchmark module: when the
    curve is shorter than ``WARMUP`` (e.g. the N=3 ``--smoke`` output) fall
    back to the whole curve instead of slicing to empty.
    """
    warmup = WARMUP if len(reserved_bytes) > WARMUP else 0
    return float(max(reserved_bytes[warmup:]) / GiB)


def _control_gap_gib(old_reserved, hook_reserved) -> float:
    """Reserved high-water gap (GiB): old-arm max minus hook-arm max, post-warmup."""
    return _max_reserved_gib(old_reserved) - _max_reserved_gib(hook_reserved)


def _has_positive_trend(old_reserved, hook_reserved) -> bool:
    """True iff the old arm climbs (last-30 slope) and the hook arm does not.

    The positive control's OR branch (plan §6.2 criterion 2): even when the
    retained high-water gap is small, a monotone climb on the old arm that the
    hook arm lacks demonstrates the fix removed a real growth.
    """
    old_tail = np.asarray(old_reserved[-30:], float) / GiB
    hook_tail = np.asarray(hook_reserved[-30:], float) / GiB
    old_slope = float(np.polyfit(np.arange(len(old_tail)), old_tail, 1)[0])
    hook_slope = float(np.polyfit(np.arange(len(hook_tail)), hook_tail, 1)[0])
    return old_slope >= SLOPE_TOL_GiB_per_iter and abs(hook_slope) < SLOPE_TOL_GiB_per_iter


def evaluate(expandable: dict, default: dict) -> tuple[str, str]:
    """Apply the plan §6.2 PASS / INCONCLUSIVE / REGRESSION rule set.

    Returns ``(verdict, message)``. Raises ``AssertionError`` (via
    ``_assert_grad_disabled`` / ``_assert_allocator_tag``) if either arm in
    either regime ran grad-enabled or if the JSONs were swapped between
    --expandable-json / --default-json.
    """
    _assert_allocator_tag(expandable, "expandable_segments_on")
    _assert_allocator_tag(default, "default_allocator")
    _assert_grad_disabled(expandable, "expandable_segments_on")
    _assert_grad_disabled(default, "default_allocator")

    hook_exp = flat(expandable["arms"]["hook"]["reserved"])
    hook_def = flat(default["arms"]["hook"]["reserved"])

    # Criterion 1: hook arm flat under BOTH allocators. A non-flat hook arm is a
    # #671 regression (REGRESSION takes precedence over everything).
    if not hook_exp["flat"] or not hook_def["flat"]:
        offenders = []
        if not hook_exp["flat"]:
            offenders.append(
                f"expandable(span={hook_exp['span_GiB']:.3f} GiB, "
                f"slope={hook_exp['tail_slope_GiB_per_iter']:.4f}/iter)"
            )
        if not hook_def["flat"]:
            offenders.append(
                f"default(span={hook_def['span_GiB']:.3f} GiB, "
                f"slope={hook_def['tail_slope_GiB_per_iter']:.4f}/iter)"
            )
        return (
            REGRESSION,
            "hook arm is NOT flat (#671 regression): "
            + "; ".join(offenders)
            + f" — exceeds ABS_TOL_GiB={ABS_TOL_GiB} / SLOPE_TOL={SLOPE_TOL_GiB_per_iter}",
        )

    # Criterion 2: positive control discriminates under expandable_segments:True.
    old_exp_reserved = expandable["arms"]["old_ohs_true"]["reserved"]
    hook_exp_reserved = expandable["arms"]["hook"]["reserved"]
    gap = _control_gap_gib(old_exp_reserved, hook_exp_reserved)
    trend = _has_positive_trend(old_exp_reserved, hook_exp_reserved)

    if gap >= CTRL_GAP_GiB or trend:
        how = (
            f"reserved-gap={gap:.4f} GiB >= CTRL_GAP_GiB={CTRL_GAP_GiB}"
            if gap >= CTRL_GAP_GiB
            else f"old-arm positive monotone trend (reserved-gap={gap:.4f} GiB)"
        )
        return (
            PASS,
            f"hook arm flat under both allocators; positive control discriminates ({how}).",
        )

    return (
        INCONCLUSIVE,
        f"hook arm flat under both allocators, but the positive control did NOT "
        f"discriminate at this scale (reserved-gap={gap:.4f} GiB < "
        f"CTRL_GAP_GiB={CTRL_GAP_GiB}, no positive monotone trend). The benchmark at "
        f"N=50 / one model copy did not exhibit the #545 growth; NOT a false PASS "
        f"(see plan §6.2 / §7 — escalate to 2-copy / N=100 or record inconclusive).",
    )


def _plot_regime(results: dict, stem: str) -> None:
    """Save a reserved+allocated line PNG (hook vs old) for one allocator regime."""
    import matplotlib.pyplot as plt

    set_paper_style("blog")
    fig, ax = plt.subplots()
    arms = results["arms"]
    n = len(arms["hook"]["reserved"])
    x = np.arange(n)
    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    ax.plot(
        x, np.asarray(arms["hook"]["reserved"], float) / GiB, color=primary, label="hook reserved"
    )
    ax.plot(
        x,
        np.asarray(arms["hook"]["allocated"], float) / GiB,
        color=primary,
        linestyle="--",
        label="hook allocated",
    )
    ax.plot(
        x,
        np.asarray(arms["old_ohs_true"]["reserved"], float) / GiB,
        color=baseline,
        label="old output_hidden_states reserved",
    )
    ax.plot(
        x,
        np.asarray(arms["old_ohs_true"]["allocated"], float) / GiB,
        color=baseline,
        linestyle="--",
        label="old output_hidden_states allocated",
    )
    ax.set_xlabel("Extraction iteration")
    ax.set_ylabel("GPU memory (GiB)")
    ax.legend()
    savefig_paper(fig, stem, dir="figures/issue_673")
    plt.close(fig)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expandable-json",
        type=Path,
        required=True,
        help="Path to the expandable_segments:True regime JSON.",
    )
    parser.add_argument(
        "--default-json",
        type=Path,
        required=True,
        help="Path to the default-allocator regime JSON.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG generation (verdict only).",
    )
    args = parser.parse_args(argv)

    expandable = _load(args.expandable_json)
    default = _load(args.default_json)

    verdict, message = evaluate(expandable, default)

    if not args.no_figures:
        _plot_regime(expandable, "memory_vs_iter_expandable_segments_on")
        _plot_regime(default, "memory_vs_iter_default_allocator")

    print(f"{verdict}: {message}")
    return 1 if verdict == REGRESSION else 0


if __name__ == "__main__":
    sys.exit(main())
