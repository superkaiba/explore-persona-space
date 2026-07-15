"""#1092 inline free-analysis — FIT-FREE battery-caveat repairs.

Delivers the repairs that do NOT require a ridge refit (Step-0 verdict: the
battery-excluded READ1 refit + per-target R² + Part-B operator maps project to
>12 h on the shared VM — the production grid was ~230 CPU machine-h across 12
GCE n2-highmem-16 boxes — so per the dispatch's Step-0 gate the refit battery is
NOT launched here and is reported for off-VM dispatch).

What IS computed here (all fit-free, reusing the committed engine helpers so
numbers are byte-comparable to the banked reads):

  A1-shares/additivity : read3 (anova variance shares) + read4 (operator
      additivity residuals) recomputed battery-EXCLUDED and shown identical to
      the banked (battery-included) values — because both are dense-core-only
      statistics and the dense core (manifest positions 0..4751) is disjoint
      from the battery block (18793..21192).
  A1-scope             : which cells the battery deviation actually touches
      (only the 4 full-corpus cells; the claude/shuf cells capture a
      battery-free manifest prefix).
  A3-floors            : per-target affine/identity/train-mean transport floors
      (the banked pooled-ambient floors were NaN because the stacked 10752-d
      target != the 3584-d input; per single target they are same-dim and
      defined). Battery-excluded fit folds.

Deferred to an off-VM refit (documented in the return, not run here):
  read1 battery-excluded held-out R² for the 4 affected cells, A2 per-target
  R², A2 topic-matched pairing delta, monitoring-gap curve, and Part-B operator
  comparison (all require the PressRidge refit / fitted maps).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue1092_fit_grid as eng  # noqa: E402
import numpy as np  # noqa: E402

STAGE = Path("data/issue_1092_inline")
SUMM = STAGE / "issue1092_realistic_crossing/analysis_tensors/summaries"
MANIFEST = STAGE / "issue1092_realistic_crossing/corpus/manifest.jsonl"
BANKED = Path("eval_results/issue_1092/p7")
OUT = Path("eval_results/issue_1092/inline_caveat_repairs_operator_comparison")

CELLS = [
    "cell_inst_own",
    "cell_inst_claude",
    "cell_inst_pretext",
    "cell_inst_shuf",
    "cell_pre_own",
    "cell_pre_claude",
    "cell_pre_insttext",
    "cell_pre_shuf",
]
TARGETS = ("t1", "t2", "t3")
# A3 transport floors: primary target t1 only (answer-span mean) — the point
# (floors << ridge R²) is target-invariant; t2/t3 deferred with the refit.
FLOOR_TARGETS = ("t1",)
LAYER = 14


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _banked_units(fname: str) -> dict:
    with open(BANKED / fname) as fh:
        d = json.load(fh)
    out = {}
    for u in d["units"]:
        p = u["provenance"]
        out[(p["cell"], p["arm"], p["fit_arm"], p["layer"], p["basis"])] = u
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = eng._jsonl(MANIFEST)
    n_manifest = len(rows)
    print(f"manifest rows={n_manifest}", flush=True)

    read1 = _banked_units("read1_map_skill.json")
    read3 = _banked_units("read3_fgi_shares.json")
    read4 = _banked_units("read4_operator_identity.json")

    result: dict = {
        "meta": {
            "script": "scripts/issue1092_inline_repairs.py",
            "git_commit": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "manifest_rows": n_manifest,
            "manifest_fingerprint": "7ef5523673d6",
            "note": (
                "Fit-free battery-caveat repairs; the battery-excluded ridge refit "
                "(read1 R², per-target R², Part-B maps) is DEFERRED to an off-VM CPU "
                "box per the Step-0 >4h gate."
            ),
        }
    }

    # ---- A1-scope: which cells does the battery deviation actually touch? ----
    bat = [i for i, r in enumerate(rows) if r.get("is_eval_only")]
    tr = [i for i, r in enumerate(rows) if r.get("stratum") == "trait_stratum"]
    dc = [i for i, r in enumerate(rows) if r.get("stratum") == "dense_core"]
    scope = {
        "battery_positions": {"min": min(bat), "max": max(bat), "n": len(bat)},
        "trait_positions": {"min": min(tr), "max": max(tr), "n": len(tr)},
        "dense_core_positions": {"min": min(dc), "max": max(dc), "n": len(dc)},
        "dense_core_disjoint_from_battery": max(dc) < min(bat),
        "battery_stratum_label": "battery",
        "engine_filter_excludes": ["trait_stratum", "battery_eval_only"],
        "engine_filter_is_noop": "battery rows are labelled 'battery' not 'battery_eval_only'",
        "correct_exclusion_key": "is_eval_only == True (== stratum 'battery', n=2400)",
        "cells": {},
    }
    for cell in CELLS:
        n0 = int(np.load(SUMM / cell / f"context_end_L{LAYER:02d}.npy", mmap_mode="r").shape[0])
        br = [i for i in bat if i < n0]
        trr = [i for i in tr if i < n0]
        scope["cells"][cell] = {
            "n0_captured": n0,
            "battery_in_fit_range": len(br),
            "fitA_banked_n": n0 - len(trr),
            "fitA_battery_excluded_n": n0 - len(trr) - len(br),
            "battery_affected": len(br) > 0,
        }
    result["A1_scope"] = scope
    print(
        "A1-scope: affected cells =",
        [c for c, v in scope["cells"].items() if v["battery_affected"]],
        flush=True,
    )

    # ---- A1: read1 banked (OLD) headline table @ L14 ambient fit-arm A ----
    read1_old = {}
    for cell in CELLS:
        row = {}
        for arm in ("prefix_end", "context_end"):
            u = read1.get((cell, arm, "A", LAYER, "ambient"))
            if u:
                row[arm] = {
                    "r2": u["r2"],
                    "train_mean_floor": u["identity_floors"]["train_mean"]["mean"],
                    "perm_null_p95": u["perm_null"]["p95"],
                    "n_rows_banked": u["provenance"].get("n"),
                }
        read1_old[cell] = row
    result["A1_read1_banked_old"] = {
        "config": "layer 14, ambient, fit-arm A, pooled t1/t2/t3 (battery-INCLUDED)",
        "cells": read1_old,
    }

    # ---- Single per-cell pass: read3/read4 battery-invariance + A3 floors ----
    # Load each cell's 5 summary files ONCE (IO-bound), write incrementally.
    print("\nPer-cell pass (read3/read4 invariance + A3 floors) ...", flush=True)
    inv: dict = {}
    floors_out: dict = {}
    # resume: seed from a prior partial JSON so a relaunch only finishes remaining cells
    _prev = OUT / "fit_free_repairs.json"
    if _prev.exists():
        try:
            with open(_prev) as fh:
                pj = json.load(fh)
            inv.update(pj.get("A1_read3_read4_invariance", {}))
            floors_out.update(pj.get("A3_transport_floors", {}).get("cells", {}))
            print(f"resume: {len(floors_out)} cells already done", flush=True)
        except Exception:
            pass
    result["A1_read3_read4_invariance"] = inv
    result["A3_transport_floors"] = {
        "config": "layer 14, ambient, fit-arm A battery-EXCLUDED, target t1 (answer-span mean)",
        "note": (
            "identity/affine floors are same-dim only per single 3584-d target (t1); the banked "
            "pooled floors were NaN (stacked 10752-d target != 3584-d input) — the flagged engine "
            "gap. Ridge reference is the banked POOLED R² (battery-INCLUDED); the per-target "
            "battery-excluded ridge is deferred with the refit. t2/t3 floors deferred "
            "(target-invariant point)."
        ),
        "cells": floors_out,
    }
    # dense_core occupies manifest positions [0..4751] exactly (verified); read3/read4
    # are dense-core-only statistics, so recompute them on the dense-core slice via mmap
    # (instant, battery-irrelevant) rather than loading the full 17k-row fp64 arrays.
    dc_rows = [r for r in rows if r.get("stratum") == "dense_core"]
    n_dc = len(dc_rows)
    assert all(rows[i].get("stratum") == "dense_core" for i in range(n_dc)), (
        "dense_core not a prefix"
    )

    def _dc(cell: str, kind: str) -> np.ndarray:
        p = eng._summary_shard_paths(SUMM, cell, kind, LAYER)[0]
        return np.load(p, mmap_mode="r")[:n_dc].astype(np.float64)

    for cell in CELLS:
        if cell in floors_out and cell in inv:
            continue
        prefix = eng._load_summary(SUMM, cell, "prefix_end", LAYER)[0]
        ctx = eng._load_summary(SUMM, cell, "context_end", LAYER)[0]
        t1_full = eng._load_summary(SUMM, cell, "t1", LAYER)[0]
        n0 = ctx.shape[0]
        base = rows[:n0]
        idx_out = np.asarray(
            [
                i
                for i, r in enumerate(base)
                if r.get("stratum") != "trait_stratum" and not r.get("is_eval_only")
            ],
            dtype=np.int64,
        )
        # --- read3/read4 battery-excluded (dense-core slice) vs banked (battery-included) ---
        Yb_dc = np.concatenate([t1_full[:n_dc], _dc(cell, "t2"), _dc(cell, "t3")], axis=1)
        sh = eng._anova_shares(dc_rows, Yb_dc)
        r4 = eng._operator_identity_read(dc_rows, ctx[:n_dc], Yb_dc, seed=LAYER, n_draws=0)
        b3 = read3.get((cell, "context_end", "A", LAYER, "ambient"), {}).get("anova_shares", {})
        b4 = read4.get((cell, "context_end", "A", LAYER, "ambient"), {}).get(
            "operator_identity", {}
        )
        inv[cell] = {
            "read3_shares_battery_excluded": {
                "prefix": sh["share_prefix"],
                "query": sh["share_query"],
                "interaction": sh["share_interaction"],
                "n_dense_core": sh["n_rows"],
            },
            "read3_shares_banked": {
                "prefix": b3.get("share_prefix"),
                "query": b3.get("share_query"),
                "interaction": b3.get("share_interaction"),
            },
            "read4_residuals_battery_excluded": {
                "residual_interaction_over_total": r4["residual_interaction_norm_over_total"],
                "mprime_minus_m_minus_g_over_g": r4["mprime_minus_m_minus_g_over_g"],
            },
            "read4_residuals_banked": {
                "residual_interaction_over_total": b4.get("residual_interaction_norm_over_total"),
                "mprime_minus_m_minus_g_over_g": b4.get("mprime_minus_m_minus_g_over_g"),
                "null_p05": (b4.get("random_map_pairing_null") or {}).get("p05"),
            },
            "shares_max_abs_delta": max(
                abs(sh["share_prefix"] - (b3.get("share_prefix") or sh["share_prefix"])),
                abs(sh["share_query"] - (b3.get("share_query") or sh["share_query"])),
                abs(
                    sh["share_interaction"]
                    - (b3.get("share_interaction") or sh["share_interaction"])
                ),
            ),
        }
        # --- A3 transport floors on t1 (battery-excluded folds) ---
        ur = [base[i] for i in idx_out]
        folds = eng._folds_from_manifest(ur, len(ur), group_key="prefix_id", n_folds=6)
        y_floor = {"t1": t1_full[idx_out]}
        Xarm = {"prefix_end": prefix[idx_out], "context_end": ctx[idx_out]}
        cell_out = {}
        for arm in ("prefix_end", "context_end"):
            bu = read1.get((cell, arm, "A", LAYER, "ambient"))
            arm_out = {
                "banked_pooled_ridge_r2_batteryIN": bu["r2"] if bu else None,
                "per_target": {},
            }
            for tg in FLOOR_TARGETS:
                fl = eng._identity_floors(Xarm[arm], y_floor[tg], folds)
                arm_out["per_target"][tg] = {
                    "train_mean": fl["train_mean"]["mean"],
                    "raw_identity": fl["raw_identity"]["mean"],
                    "global_affine_scaled_identity": fl["global_affine"]["mean"],
                    "diag_affine": fl["diag_affine"]["mean"],
                }
            cell_out[arm] = arm_out
        floors_out[cell] = cell_out
        ctx_t1 = cell_out["context_end"]["per_target"]["t1"]
        print(
            f"  {cell:20s} sharesΔ={inv[cell]['shares_max_abs_delta']:.1e} "
            f"resid_excl={r4['residual_interaction_norm_over_total']:.5f} "
            f"| ctx->t1 id={ctx_t1['raw_identity']:.3f} "
            f"aI={ctx_t1['global_affine_scaled_identity']:.3f} "
            f"diag={ctx_t1['diag_affine']:.3f} tm={ctx_t1['train_mean']:.4f}",
            flush=True,
        )
        # incremental write after each cell
        (OUT / "fit_free_repairs.json").write_text(json.dumps(result, indent=2, allow_nan=True))

    print(f"\nwrote {OUT / 'fit_free_repairs.json'}", flush=True)


if __name__ == "__main__":
    t = time.monotonic()
    main()
    print(f"done in {time.monotonic() - t:.0f}s")
