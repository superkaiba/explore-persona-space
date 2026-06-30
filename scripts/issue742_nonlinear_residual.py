#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002
"""Issue #742 Stage 1 — is *any* headroom nonlinearly extractable? (plan v7 §4 Stage 1).

Per qualifying behavior × genre (Stage 0 routed it to ``headroom``):

  0. **Synthetic dCor power pre-check + adaptive d_eff selection** (§4 Stage-1 step 0).
     If ≥0.8 power at d_eff=10, proceed; else relax d_eff (≤20, PCA-variance ≥80%)
     and pick the largest clearing dim — and if NONE clears, report the null as
     ``variance-limited`` (indistinguishable-from-null given variance), NEVER as a
     no-nonlinear-signal claim.
  1-3. **ONE PCA basis + ONE LEACE eraser on all 50 contexts** (a single
     commensurable frame, MF3), dCor on the pooled single-frame LEACE-erased points,
     with the WHOLE PCA→LEACE→dCor pipeline REFIT per permutation (Option A).
  4. **Control-task / shuffled-E0 null** (Hewitt-Liang) — the same refit-per-draw
     procedure on a fixed shuffled-label assignment.
  7. **Held-out post-LEACE linear-leakage diagnostic** (§13 NEW, Phase-2 REVISE):
     distinguish a genuine nonlinear residual from leftover LINEAR leakage LEACE's
     train-fold-only guarantee does not erase out-of-sample. The Stage-1 verdict is
     classified from (dcor_pass, linear_pass).

CPU-only. Writes ``eval_results/issue_742/stage1_leace_dcor.json`` (absent-by-design
when no behavior×genre has wide bracket; then stage0 records all-ceiling-limited).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue404_common import reproducibility_metadata  # noqa: E402

from explore_persona_space.analysis import issue_742_decoding_ceiling as dc  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_658"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_742"


def _select_layer(genre: str, behavior: str) -> int:
    """Return #658's A3.3 best layer for (behavior, genre) from analyzer_body_data.json.

    BLOCKER-fix stage1-routing-layer(b): the A3.3 read-out layers are PER-BEHAVIOR ×
    PER-GENRE and live at ``/<genre>/a33/<behavior>/layer`` in
    ``analyzer_body_data.json`` (e.g. Betley sycophancy → 27, refusal → 8; UltraChat
    refusal → 6) — NOT in ``locked_recipe.json``, which has no ``per_behavior`` key, so
    the prior fallback to layer 21 silently read the wrong layer for every cell. Raises
    (no silent default) when the key is absent.
    """
    return dc.load_a33_layer(behavior, genre, eval_dir=EVAL_DIR)


def _e0_vector(e0: dict, behavior: str, context_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    present = [c for c in context_ids if c in e0 and behavior in e0[c]]
    vec = np.array([float(e0[c][behavior]["rate"]) for c in present])
    return vec, present


def run_stage1_cell(
    v0_layer: np.ndarray,
    E0: np.ndarray,
    *,
    behavior: str,
    genre: str,
    layer: int,
    d_eff: int,
    n_perm: int,
    rng: np.random.Generator,
    run_power_check: bool,
    power_trials: int,
    power_perm: int,
) -> dict:
    """One Stage-1 cell: power pre-check → PCA→LEACE→dCor + control-task + held-out."""
    n = v0_layer.shape[0]

    # step 0 — adaptive d_eff power selection (§4 Stage-1 step 0)
    power_sel: dict | None = None
    chosen_d_eff = d_eff
    if run_power_check:

        def _var_retained(dd: int) -> float:
            basis = dc.fit_pca_basis(v0_layer, dd)
            return float(np.sum(basis.explained_variance_ratio))

        sel = dc.select_d_eff_for_power(
            candidates=(d_eff, 15, 20),
            target_power=0.8,
            n=n,
            n_perm=power_perm,
            effect=0.10,
            n_trials=power_trials,
            rng=rng,
            variance_retained_fn=_var_retained,
            variance_floor=0.80,
        )
        chosen_d_eff = sel.chosen_d_eff
        power_sel = {
            "chosen_d_eff": sel.chosen_d_eff,
            "realized_power": sel.realized_power,
            "variance_limited": sel.variance_limited,
            "target_power": sel.target_power,
            "per_d_eff_power": {str(k): v for k, v in sel.per_d_eff_power.items()},
        }

    # steps 1-3 — single full-sample PCA→LEACE→dCor, refit per permutation
    dcor_res = dc.dcor_permutation_test(v0_layer, E0, d_eff=chosen_d_eff, n_perm=n_perm, rng=rng)
    dcor_pass = bool(dcor_res.p_value < 0.05)

    # step 4 — control-task / shuffled-E0 null (Hewitt-Liang): one fixed shuffle, then
    # the same refit-per-draw permutation test on the shuffled labels.
    shuffled = E0[rng.permutation(n)]
    control_res = dc.dcor_permutation_test(
        v0_layer, shuffled, d_eff=chosen_d_eff, n_perm=n_perm, rng=rng
    )

    # step 7 — held-out post-LEACE linear-leakage diagnostic (§13 NEW). Split 50 into
    # train/held-out; LEACE on PCA-reduced train, probe held-out residual for residual
    # linear E0-correlation against a held-out permutation null.
    basis = dc.fit_pca_basis(v0_layer, chosen_d_eff)
    reduced = basis.transform(v0_layer)
    perm = rng.permutation(n)
    n_train = round(0.7 * n)
    tr, he = perm[:n_train], perm[n_train:]
    held = dc.held_out_linear_leakage(reduced[tr], E0[tr], reduced[he], E0[he], rng=rng)
    linear_pass = bool(held.post_leace_linear_pass)

    verdict = dc.classify_stage1_verdict(dcor_pass=dcor_pass, linear_pass=linear_pass)
    # a variance-limited power read forbids a no-signal claim: a non-significant dCor
    # under variance-limited power reads "indistinguishable-from-null given variance"
    if power_sel is not None and power_sel["variance_limited"] and not dcor_pass:
        verdict = "indistinguishable-from-null-given-variance"

    return {
        "behavior": behavior,
        "genre": genre,
        "layer": layer,
        "d_eff": chosen_d_eff,
        "coordinate_scheme": "single_full_sample_pca_basis+single_full_sample_leace; "
        "dCor pipeline refit per permutation (Option A)",
        "power_selection": power_sel,
        "dcor_observed": dcor_res.dcor,
        "dcor_p_value": dcor_res.p_value,
        "dcor_null_median": float(np.median(dcor_res.null)),
        "dcor_pass": dcor_pass,
        "control_task_dcor": control_res.dcor,
        "control_task_p_value": control_res.p_value,
        "held_out_linear_rho": held.rho,
        "held_out_null_ci": list(held.null_ci),
        "post_leace_linear_pass": linear_pass,
        "stage1_verdict": verdict,
        "n_perm": n_perm,
    }


def run(
    *,
    behaviors: list[str],
    genres: list[str],
    d_eff: int,
    n_perm: int,
    seed: int,
    stage0_path: Path | None,
    power_trials: int,
    power_perm: int,
    force_all: bool,
) -> dict:
    rng = np.random.default_rng(seed)

    # BLOCKER-fix stage1-routing-layer(a): respect the Stage-0 internal gate. Stage 1
    # runs ONLY on cells Stage-0 routed to `headroom`. When a Stage-0 output exists and
    # reports EVERY read-out cell ceiling-limited, Stage 1 does NOT fire — it writes a
    # `fired: false` stub (the plan's absent-by-design contract) instead of running the
    # dCor test on cells with no headroom to test (the prior round's gate violation:
    # `not qualifying` silently fell through to running every cell).
    stage0_seen = False
    qualifying: set[tuple[str, str]] = set()
    if not force_all and stage0_path is not None and stage0_path.exists():
        stage0_seen = True
        s0 = json.loads(stage0_path.read_text())
        for e in s0.get("brackets", []):
            if e.get("gate_verdict") == "headroom":
                qualifying.add((e["behavior"], e["genre"]))
    if force_all:
        # explicit override (smoke / robustness): run every requested cell
        qualifying = {(b, g) for g in genres for b in behaviors}
    elif stage0_seen and not qualifying:
        # Stage-0 ran and found NO headroom anywhere -> Stage 1 does not fire.
        return {
            "task": "issue_742",
            "stage": "stage1_leace_dcor",
            "cells": [],
            "fired": False,
            "absent_by_design": True,
            "reason": "all read-out behaviors ceiling-limited at Stage 0 "
            "(no cell routed to headroom; the internal gate skips Stage 1)",
            "config": {
                "behaviors": behaviors,
                "genres": genres,
                "d_eff": d_eff,
                "n_perm": n_perm,
                "seed": seed,
                "power_trials": power_trials,
                "power_perm": power_perm,
            },
            "metadata": reproducibility_metadata({"script": "issue742_nonlinear_residual"}),
        }
    elif not stage0_seen:
        # no Stage-0 output to gate on (e.g. standalone invocation) -> run all cells,
        # flagged so the absence of a gate is auditable (NOT a silent all-cells run).
        qualifying = {(b, g) for g in genres for b in behaviors}

    results: list[dict] = []
    for genre in genres:
        gi = dc.load_inputs(genre, repo_root=PROJECT_ROOT)
        for behavior in behaviors:
            if (behavior, genre) not in qualifying:
                continue
            layer = _select_layer(genre, behavior)
            V = dc.stack_v0(gi.v0_dict, recipe="last")  # (n, n_layers, d)
            v0_layer = V[:, layer, :]
            E0, present = _e0_vector(gi.E0_per_behavior, behavior, gi.context_ids)
            v0_layer = v0_layer[: len(present)]
            entry = run_stage1_cell(
                v0_layer,
                E0,
                behavior=behavior,
                genre=genre,
                layer=layer,
                d_eff=d_eff,
                n_perm=n_perm,
                rng=rng,
                run_power_check=True,
                power_trials=power_trials,
                power_perm=power_perm,
            )
            results.append(entry)

    return {
        "task": "issue_742",
        "stage": "stage1_leace_dcor",
        "cells": results,
        "fired": len(results) > 0,
        "absent_by_design": len(results) == 0,
        "gate_source": (
            "force_all" if force_all else ("stage0_brackets" if stage0_seen else "no_stage0_gate")
        ),
        "config": {
            "behaviors": behaviors,
            "genres": genres,
            "d_eff": d_eff,
            "n_perm": n_perm,
            "seed": seed,
            "power_trials": power_trials,
            "power_perm": power_perm,
        },
        "metadata": reproducibility_metadata({"script": "issue742_nonlinear_residual"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #742 Stage 1: PCA+LEACE+dCor nonlinear-residual test."
    )
    parser.add_argument("--behaviors", default=",".join(dc.READOUT_BEHAVIORS))
    parser.add_argument("--genres", default="betley,ultrachat")
    parser.add_argument("--d-eff", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=742)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--power-perm", type=int, default=1000)
    parser.add_argument(
        "--stage0",
        type=Path,
        default=OUT_DIR / "stage0_brackets.json",
        help="Stage-0 output to read gate verdicts from (headroom cells only)",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="run Stage 1 on every behavior×genre regardless of the Stage-0 gate",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    d_eff, n_perm = args.d_eff, args.n_perm
    power_trials, power_perm = args.power_trials, args.power_perm
    force_all = args.force_all
    stage0 = args.stage0
    if args.smoke:
        behaviors = behaviors[:1]
        genres = genres[:1]
        n_perm = 3
        power_trials = 3
        power_perm = 10
        force_all = True
        stage0 = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        behaviors=behaviors,
        genres=genres,
        d_eff=d_eff,
        n_perm=n_perm,
        seed=args.seed,
        stage0_path=stage0,
        power_trials=power_trials,
        power_perm=power_perm,
        force_all=force_all,
    )
    out_path = args.out_dir / (
        "stage1_leace_dcor_smoke.json" if args.smoke else "stage1_leace_dcor.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage1_leace_dcor] wrote {out_path} ({len(result['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
