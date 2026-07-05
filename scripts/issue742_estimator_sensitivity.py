# ruff: noqa: RUF001, RUF002, RUF003
"""Stage-0 headline-estimator sensitivity re-read (issue #742, Step 9a-ter follow-up).

Recomputes the 8 Stage-0 gate verdicts with the SPLIT-HALF reliability ceiling as
the headline estimator instead of the production BINOMIAL headline, to test whether
any gate verdict is an artifact of the estimator choice. Pure re-aggregation of the
committed ``eval_results/issue_742/stage0_brackets.json`` — NO new data, NO judge
calls, NO GPU.

Estimator caveat (why the split-half read is POINT-based, not CI-based):
    The production gate (``issue742_reliability.stage0_gate_verdict``) is a
    CI-membership test — ρ_lin vs the CLUSTER-BOOTSTRAP CI on √(r_yy). That
    bootstrap re-derives the binomial statistic per resample from the raw
    per-context (rate, m_cell) probe matrix, which lives in ``E0_expression.json``
    and is NOT in the committed eval_results tree. So a split-half cluster-bootstrap
    CI CANNOT be re-derived here honestly. This sensitivity re-read therefore swaps
    the headline POINT (binomial → split-half) and applies the SAME canonical gate
    rule at that point, which is exactly the degenerate ``[point, point]`` CI case of
    ``stage0_gate_verdict`` (ρ_lin < point ⇒ headroom, else ceiling-limited). The
    production BINOMIAL verdict is re-read the same POINT way for an apples-to-apples
    flip comparison, and the production CI verdict is carried alongside + asserted to
    reproduce the stored ``gate_verdict`` (self-consistency check).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Root on this script's own tree (the issue-742 worktree/branch), NOT
# task_workflow.repo_root(): the stage0 JSONs are committed on the issue
# branch and absent from the main checkout that repo_root() resolves.
_TREE_ROOT = Path(__file__).resolve().parents[1]

if str(_TREE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_TREE_ROOT / "scripts"))

from issue742_reliability import stage0_gate_verdict  # noqa: E402


def repo_root() -> Path:
    """Return the git tree this script lives in (worktree-safe)."""
    return _TREE_ROOT


EVAL_DIR = repo_root() / "eval_results" / "issue_742"
BRACKETS_PATH = EVAL_DIR / "stage0_brackets.json"
OUT_PATH = EVAL_DIR / "stage0_gate_sensitivity.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def _point_verdict(rho_lin: float, point: float) -> str:
    """Gate rule at a single √(r_yy) point (degenerate [point, point] CI).

    Reuses the CANONICAL ``stage0_gate_verdict`` so the point read is NOT a
    reimplementation of the verdict rule — ρ_lin < point ⇒ ``"headroom"``.
    """
    return stage0_gate_verdict(rho_lin, point, point)


def main() -> None:
    data = json.loads(BRACKETS_PATH.read_text())
    brackets = data["brackets"]

    cells = []
    flips = 0
    for b in brackets:
        rho = float(b["rho_lin"])
        sqrt_binom = float(b["sqrt_r_yy_binomial"])
        sqrt_sh = b["sqrt_r_yy_split_half"]
        ci_lo, ci_hi = (float(b["sqrt_r_yy_ci"][0]), float(b["sqrt_r_yy_ci"][1]))
        stored_verdict = b["gate_verdict"]

        # Self-consistency: the canonical gate rule on the stored binomial cluster-
        # bootstrap CI MUST reproduce the persisted production verdict.
        prod_ci_verdict = stage0_gate_verdict(rho, ci_lo, ci_hi)
        assert prod_ci_verdict == stored_verdict, (
            f"{b['behavior']}/{b['genre']}: re-derived production CI verdict "
            f"{prod_ci_verdict!r} != stored {stored_verdict!r} — gate-rule/data drift"
        )

        # Apples-to-apples POINT reads (same canonical rule, degenerate CI).
        binom_point_verdict = _point_verdict(rho, sqrt_binom)
        if sqrt_sh is None:
            split_half_verdict = None
            flips_flag = None
        else:
            split_half_verdict = _point_verdict(rho, float(sqrt_sh))
            flips_flag = split_half_verdict != binom_point_verdict
            if flips_flag:
                flips += 1

        cells.append(
            {
                "behavior": b["behavior"],
                "genre": b["genre"],
                "rho_lin": rho,
                # headline points
                "sqrt_r_yy_binomial": sqrt_binom,
                "sqrt_r_yy_split_half": None if sqrt_sh is None else float(sqrt_sh),
                # production CI verdict (binomial cluster-bootstrap) — the persisted headline
                "sqrt_r_yy_ci": [ci_lo, ci_hi],
                "binomial_ci_verdict": prod_ci_verdict,
                # apples-to-apples POINT verdicts under each estimator
                "binomial_point_verdict": binom_point_verdict,
                "split_half_verdict": split_half_verdict,
                "verdict_flips": flips_flag,
                # bracket widths under each headline estimator (√(r_yy) − ρ_lin)
                "bracket_width_binomial": sqrt_binom - rho,
                "bracket_width_split_half": None if sqrt_sh is None else float(sqrt_sh) - rho,
            }
        )

    out = {
        "task": "issue_742",
        "stage": "stage0_gate_sensitivity",
        "question": (
            "Do any of the 8 Stage-0 gate verdicts flip when the SPLIT-HALF reliability "
            "ceiling is the headline estimator instead of the production BINOMIAL headline?"
        ),
        "method_note": (
            "POINT-based sensitivity re-read: swap the headline √(r_yy) point "
            "(binomial→split-half) and apply the canonical stage0_gate_verdict rule at "
            "the point (degenerate [point,point] CI ⇒ ρ_lin<point ⇒ headroom). A "
            "split-half CLUSTER-BOOTSTRAP CI is NOT re-derivable from committed files "
            "(raw per-context probe matrix lives in un-committed E0_expression.json), so "
            "the split-half headline is necessarily a point read. binomial_point_verdict "
            "is the matched-footing binomial POINT read; binomial_ci_verdict is the "
            "production CI verdict (reproduces the stored gate_verdict, asserted)."
        ),
        "n_cells": len(cells),
        "n_verdict_flips_split_half_vs_binomial_point": flips,
        "cells": cells,
        "source_brackets": str(BRACKETS_PATH.relative_to(repo_root())),
        "reproducibility": {
            "git_commit": _git_commit(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "generator": "scripts/issue742_estimator_sensitivity.py",
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.relative_to(repo_root())}")
    print(f"n_cells={len(cells)}  flips(split_half vs binomial point)={flips}")
    for c in cells:
        print(
            f"  {c['behavior']:>20}/{c['genre']:<10} "
            f"binom_point={c['binomial_point_verdict']:<15} "
            f"split_half={c['split_half_verdict']!s:<15} "
            f"flips={c['verdict_flips']}  (prod_ci={c['binomial_ci_verdict']})"
        )


if __name__ == "__main__":
    main()
