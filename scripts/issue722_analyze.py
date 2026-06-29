#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, r̂, ρ, M⁺) in scientific docstrings + log messages + string literals.
"""Issue #722 — assemble the per-(behavior, layer) cell JSONs into the 4 deliverables.

Reads the per-cell checkpoints written by ``issue722_fit_M.py`` under
``eval_results/issue_722/cells/`` and emits the plan §6.5 primary deliverables:

- ``function_change.json`` — per cell: ``Delta_med``, the three floors,
  ``floor_combined``, ``Delta_med_ci`` (clustered_bootstrap_scalar), support
  distance, large-shift-excluded Δ, and the H_function/H_input/H_mixed call.
- ``chain_rho_M0_Mplus.json`` — ρ(M0), ρ(M⁺), the difference, family-clustered
  marginal + PAIRED CIs (clustered_bootstrap_spearman + the paired ρ-shift CI),
  MLP + shuffle reads.
- ``cross_transfer.json`` — M0→v⁺, M⁺→v⁺, M⁺→v0 transfer cosines per cell.
- ``nonlinearity_gap.json`` — (ρ_MLP − ρ_ridge) under M0 vs M⁺ per cell.

Also applies the §3 kill criterion at L=14 across the headline behaviors and the
#697 cross-reference panel (N/A when #697's f_CV artifact is absent). All four
JSONs carry reproducibility metadata (git commit, timestamps).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue722.analyze")

HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")
PRIMARY_LAYER = 14


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "issue": 722,
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _call_from_ci(ci: dict | None, floor) -> str:
    """H_function / H_input / H_mixed / undetermined from one CI vs floor_combined (plan §3)."""
    if not ci or floor is None:
        return "undetermined"
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if lo is None or hi is None:
        return "undetermined"
    if lo > floor:
        return "H_function"
    if hi <= floor:
        return "H_input"
    return "H_mixed"


def _h_call(cell: dict) -> tuple[str, bool]:
    """The per-cell H call AND the large-shift-flip flag (plan §3 support-distance).

    Returns ``(call, large_shift_flip)``. ``call`` is the FULL-CI H call
    (``H_function`` / ``H_input`` / ``H_mixed`` / ``undetermined``) from
    ``Delta_med_ci`` vs ``floor_combined`` — UNLESS excluding the large-shift
    cells flips an ``H_function`` call to anything else (MF#5 / plan §3:
    "if excluding the large-shift cells flips the H_function call, the behavior
    is reported H_mixed / inconclusive, not H_function"). On a flip the call is
    DOWNGRADED to the excluded-CI's call (H_mixed/H_input/undetermined), and
    ``large_shift_flip`` is True so the analyzer can surface the robustness
    failure. When the excluded CI is absent or agrees, the full-CI call stands
    and ``large_shift_flip`` is False.
    """
    floor = cell.get("floor_combined")
    full_call = _call_from_ci(cell.get("Delta_med_ci"), floor)
    excl_ci = cell.get("Delta_med_excl_large_shift_ci")
    if excl_ci is None:
        return full_call, False
    excl_call = _call_from_ci(excl_ci, floor)
    # Only a downgrade FROM H_function matters (the §3 robustness rule); if the
    # full CI was not H_function there is nothing to flip away from.
    if full_call == "H_function" and excl_call != "H_function":
        return excl_call, True
    return full_call, False


def _load_697_fcv() -> dict | None:
    """Per-behavior f_CV from #697 if present (local eval_results or HF), else None."""
    local = PROJECT_ROOT / "eval_results/issue_697/patch"
    if local.is_dir():
        for fname in ("f_cv.json", "fcv.json", "summary.json"):
            p = local / fname
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception as e:
                    logger.warning("could not parse #697 %s (%s)", p, e)
    return None


def assemble(cells_dir: Path, out_dir: Path) -> dict:
    """Read every per-cell JSON and write the 4 deliverables. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_files = sorted(cells_dir.glob("*_L*.json"))
    if not cell_files:
        raise FileNotFoundError(f"no per-cell JSONs under {cells_dir} (run issue722_fit_M first)")
    cells = [json.loads(p.read_text()) for p in cell_files]

    function_change = {"meta": _meta(), "cells": {}}
    chain_rho = {"meta": _meta(), "cells": {}}
    cross_transfer = {"meta": _meta(), "cells": {}}
    nonlinearity_gap = {"meta": _meta(), "cells": {}}

    for cell in cells:
        key = f"{cell['behavior']}/L{cell['layer']}"
        h_call, large_shift_flip = _h_call(cell)
        function_change["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "n_cells": cell["n_cells"],
            "Delta_med": cell["Delta_med"],
            "Delta_med_ci": cell["Delta_med_ci"],
            "Delta_med_mean_ci": cell.get("Delta_med_mean_ci"),
            "Delta_med_excl_large_shift_ci": cell.get("Delta_med_excl_large_shift_ci"),
            "floor_M0_refit": cell["floor_M0_refit"],
            "floor_Mplus_refit": cell["floor_Mplus_refit"],
            "floor_shifted": cell["floor_shifted"],
            "floor_combined": cell["floor_combined"],
            "floor_sd_combined": cell.get("floor_sd_combined"),
            "Delta_over_floor_sd": cell.get("Delta_over_floor_sd"),
            "support_distance": cell.get("support_distance"),
            "n_families": cell.get("n_families"),
            "h_call": h_call,
            "large_shift_flip": large_shift_flip,
        }
        cb = cell.get("chain_rho", {})
        chain_rho["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "n_with_E": cb.get("n_with_E"),
            "rho_M0_ridge": cb.get("rho_M0_ridge"),
            "rho_Mplus_ridge": cb.get("rho_Mplus_ridge"),
            "rho_diff_ridge": cb.get("rho_diff_ridge"),
            "ci_M0_ridge": cb.get("ci_M0_ridge"),
            "ci_Mplus_ridge": cb.get("ci_Mplus_ridge"),
            "ci_diff_ridge": cb.get("ci_diff_ridge"),  # MF#4: paired ρ-shift CI
            "rho_M0_mlp": cb.get("rho_M0_mlp"),
            "rho_Mplus_mlp": cb.get("rho_Mplus_mlp"),
            "rho_M0_shuffle": cb.get("rho_M0_shuffle"),
        }
        cross_transfer["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            **cell.get("cross_transfer", {}),
        }
        nonlinearity_gap["cells"][key] = {
            "behavior": cell["behavior"],
            "layer": cell["layer"],
            "nonlin_gap_M0": cb.get("nonlin_gap_M0"),
            "nonlin_gap_Mplus": cb.get("nonlin_gap_Mplus"),
        }

    # ---- §3 kill criterion at L=14 across headline behaviors ----
    # MF#4 (Claude Major#2): a CLEAN below-floor H_input across all 3 behaviors is
    # the pre-registered POSITIVE H_input answer (plan §3 / §8: "an all-inside-floor
    # result across all 3 behaviors with a passing shuffle check IS the H_input
    # answer, not a failure"), NOT inconclusive. Only a STRADDLE (H_mixed /
    # undetermined — incl. a large-shift-flipped H_function) counts toward the kill.
    # A below-floor H_input WITHOUT a passing MLP-vs-shuffle check cannot be cleanly
    # called H_input, so it counts as a straddle too.
    evaluated = 0
    straddle_count = 0  # H_mixed / undetermined / shuffle-failed-H_input (plan §3 ">=2 → kill")
    n_h_function = 0
    n_h_input_clean = 0
    per_behavior: dict[str, str] = {}
    for behavior in HEADLINE_BEHAVIORS:
        k = f"{behavior}/L{PRIMARY_LAYER}"
        if k not in function_change["cells"]:
            continue
        evaluated += 1
        cell = function_change["cells"][k]
        call = cell["h_call"]
        cb = chain_rho["cells"].get(k, {})
        # MLP-vs-shuffle pass on M0: MLP held-out ρ must beat the shuffle null.
        mlp, shuf = cb.get("rho_M0_mlp"), cb.get("rho_M0_shuffle")
        shuffle_pass = mlp is not None and shuf is not None and mlp > shuf
        if call == "H_function":
            n_h_function += 1
            per_behavior[behavior] = "H_function"
        elif call == "H_input" and shuffle_pass:
            n_h_input_clean += 1
            per_behavior[behavior] = "H_input_clean"
        else:
            # H_mixed, undetermined, or a below-floor H_input that failed the
            # shuffle check — all straddle/under-power outcomes (plan §3).
            straddle_count += 1
            per_behavior[behavior] = "H_input_shuffle_failed" if call == "H_input" else call
    inconclusive = evaluated > 0 and straddle_count >= 2
    if inconclusive:
        outcome = "inconclusive"
    elif evaluated > 0 and n_h_input_clean == evaluated:
        outcome = "H_input_clean"  # clean below-floor across all 3 with passing shuffle (plan §8)
    elif evaluated > 0 and n_h_function == evaluated:
        outcome = "H_function_clean"  # CI entirely above floor across all 3
    else:
        outcome = "mixed"  # a mix of H_function and clean-H_input across behaviors
    function_change["kill_criterion"] = {
        "primary_layer": PRIMARY_LAYER,
        "behaviors_evaluated": evaluated,
        "n_straddle": straddle_count,
        "n_H_function": n_h_function,
        "n_H_input_clean": n_h_input_clean,
        "per_behavior": per_behavior,
        "outcome": outcome,
        "inconclusive": inconclusive,
        "note": (
            "outcome=inconclusive iff Δ_med CI straddles floor_combined (H_mixed / "
            "undetermined / shuffle-failed-H_input) for >=2 of the 3 headline behaviors "
            "at L=14; a clean below-floor H_input with a passing MLP-vs-shuffle check "
            "across all 3 is H_input_clean (the positive H_input answer), NOT "
            "inconclusive (plan §3 / §8)"
        ),
    }

    # ---- #697 cross-reference panel (non-gating) ----
    fcv = _load_697_fcv()
    function_change["cross_ref_697"] = (
        {"status": "available", "f_cv": fcv}
        if fcv is not None
        else {"status": "N/A — #697 artifact absent at analysis time"}
    )

    # MF#1 (scope caveat): the headline Δ_med is RIDGE-only by design. Plan §4.5
    # read 1 names "Δ(c) for both ridge and MLP", but plan §3 registers the
    # ridge-only path as the PRIMARY headline (MLP-validity kill → ridge-only
    # fallback "primary, not contingency") and the H_function call gates purely on
    # the ridge Δ_med vs floor_combined. The MLP path serves the chain-ρ co-primary
    # (read 2) + the nonlinearity gap (read 4) ONLY — it is never evaluated at a
    # fresh grid input (an MLP has no closed-form off-LOCO read). So read 1's "for
    # both ridge and MLP" is interpreted as ridge-headline + MLP-as-context
    # (held-out chain-ρ + nonlinearity gap), carried as a scope caveat into the
    # clean-result (reconciler: this scoping is defensible, NOT a corruption).
    function_change["meta"]["mlp_function_change_scope_caveat"] = (
        "Headline Δ_med is RIDGE-only by design — the MLP path serves the chain-ρ "
        "co-primary + nonlinearity-gap reads only. Plan §4.5 read 1's 'for both "
        "ridge and MLP' is interpreted as ridge-headline + MLP-as-context "
        "(held-out chain-ρ + nonlinearity gap); the MLP has no closed-form "
        "off-LOCO read for M(c) at a fresh grid input. Carried as a scope caveat "
        "into the clean-result."
    )

    (out_dir / "function_change.json").write_text(
        json.dumps(function_change, indent=2, default=float)
    )
    # MF#3: the co-primary deliverable filename is chain_rho_M0_Mplus.json (plan §6.5
    # glob); the old chain_rho.json would be missed by the deliverable-collection glob.
    (out_dir / "chain_rho_M0_Mplus.json").write_text(json.dumps(chain_rho, indent=2, default=float))
    (out_dir / "cross_transfer.json").write_text(
        json.dumps(cross_transfer, indent=2, default=float)
    )
    (out_dir / "nonlinearity_gap.json").write_text(
        json.dumps(nonlinearity_gap, indent=2, default=float)
    )
    logger.info(
        "[phase=analyze] wrote 4 deliverables (%d cells); kill_outcome=%s inconclusive=%s",
        len(cells),
        function_change["kill_criterion"]["outcome"],
        function_change["kill_criterion"]["inconclusive"],
    )
    return {"n_cells": len(cells), "kill": function_change["kill_criterion"]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #722 assemble fit cells → 4 deliverables")
    ap.add_argument("--cells-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722/cells")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722")
    args = ap.parse_args()
    logger.info("[phase=analyze] cells=%s out=%s", args.cells_dir, args.out_dir)
    assemble(args.cells_dir, args.out_dir)
    logger.info("[phase=analyze] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
