#!/usr/bin/env python
"""Issue #1345 conversation-paired-stories round — matched_row_refits phase.

Matched-row comparator refits (plan v8 §4, ≈zero GPU, closed-form on the
already-persisted stores via the reused `issue825_fit_cells.run_cell` core):

  1. `R_instruct_r1_matched_{arm}` / `R_instruct_r2_matched_{arm}` — the r1/r2
     within cells refit on the r4-kept conv_id subset: the SAME-N yardstick for
     the "recovers the chat ceiling" read (removes the n-confound the critics
     flagged).
  2. `R_instruct_r4_tf_on_companion_{arm}` — the r4 TF cell refit on the
     companion's exact <=200-conv subset, so the TF-vs-on-policy calibration
     gap is not mechanically driven by n differences (plan v8 §4).
  3. `tf_op_calibration.json` — the plan §7 nested-tier TF-distortion read:
     tier 1 (qualification) when companion R² is > 0.05 below the TF cell on
     the matched subset; tier 2 (TF-DISTORTED kill label) when companion R² is
     NEGATIVE and > 0.20 below. Both are REPORTING labels the analyzer carries
     into the clean-result — this phase never halts on them.

Outputs: eval_results/issue_1345/<variant>/matched_row/*.json (+ preds caches).
Under --smoke the same chain runs with the inherited degenerate-fold /
missing-cell informational skips (gate-calibration rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1345_common as c  # noqa: E402
from issue1345_fit_cells import load_matched, run_cells  # noqa: E402

MODEL = "instruct"  # base r4 cells are N/A by scope (plan v8 §5)


def matched_row_cells(r4cfg: dict) -> tuple[list[dict], dict]:
    """(cells, allowlist_by_cell_id) for the matched-row comparator refits."""
    cells: list[dict] = []
    allow: dict[str, list[str]] = {}
    for arm in c.ARMS:
        for regime in ("r1", "r2"):
            cell = {
                "cell_id": f"R_{c.MODEL_SLUG[MODEL]}_{regime}_matched_{arm}",
                "model_key": MODEL,
                "format_key": c.REGIME_FORMAT[regime],
                "track": c.TRACK,
                "slot_index": c.ARM_SLOT_INDEX[arm],
                "target_turn_index": c.TARGET_TURN_INDEX[regime],
                "regime": regime,
                "arm": arm,
            }
            cells.append(cell)
            allow[cell["cell_id"]] = r4cfg["r4_convs"]
        if r4cfg.get("op_companion_convs"):
            cell = {
                "cell_id": f"R_{c.MODEL_SLUG[MODEL]}_r4_tf_on_companion_{arm}",
                "model_key": MODEL,
                "format_key": c.REGIME_FORMAT["r4"],
                "track": c.TRACK,
                "slot_index": c.ARM_SLOT_INDEX[arm],
                "target_turn_index": c.TARGET_TURN_INDEX["r4"],
                "regime": "r4",
                "arm": arm,
            }
            cells.append(cell)
            allow[cell["cell_id"]] = r4cfg["op_companion_convs"]
    return cells, allow


def _l19_reads(path: Path) -> dict | None:
    """{r2, ci_lo, ci_hi} at L19 from a cells JSON (None when absent)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {"r2_l19": float(d["r2_per_layer_obs"][19]), "n_rows": d.get("n_rows")}
    boot = d.get("r2_bootstrap_ci_frozen_layers_conv", {}).get("19")
    if boot:
        out.update({"ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "n_groups": boot["n_groups"]})
    return out


def tf_op_calibration(
    eval_dir: Path, matched_out: Path, *, smoke: bool, companion_halted: bool = False
) -> None:
    """Plan v8 §7 nested-tier TF-distortion read (context arm, instruct, L19).

    Reads: the full TF cell + the companion cell (fits phase, eval_dir) and the
    TF-on-companion-subset refit (this phase). Tiers are computed on the
    MATCHED companion subset (gap not mechanically n-driven). Reporting-only —
    never a process halt; under smoke missing cells skip informationally.
    ``companion_halted`` (from the matched record's ``companion`` demotion,
    i.e. no usable companion convs — the rc=23 lane) is the ONLY production
    excuse for missing companion cells; it writes an explicit
    ``companion: halted`` record instead of tiers (plan v8 §4.5).
    """
    reads = {
        "tf_full": _l19_reads(eval_dir / f"cells_{c.cell_id(MODEL, 'r4', 'context')}.json"),
        "op_companion": _l19_reads(eval_dir / f"cells_{c.cell_id(MODEL, 'r4op', 'context')}.json"),
        "tf_on_companion_subset": _l19_reads(
            matched_out / f"cells_R_{c.MODEL_SLUG[MODEL]}_r4_tf_on_companion_context.json"
        ),
        "r1_matched": _l19_reads(
            matched_out / f"cells_R_{c.MODEL_SLUG[MODEL]}_r1_matched_context.json"
        ),
        "r1_full": _l19_reads(eval_dir / f"cells_{c.cell_id(MODEL, 'r1', 'context')}.json"),
        "r2_matched": _l19_reads(
            matched_out / f"cells_R_{c.MODEL_SLUG[MODEL]}_r2_matched_context.json"
        ),
    }
    payload: dict = {
        "metadata": c.metadata(fc.FIT_SEED, 0, "scripts/issue1345_matched_row_refits.py"),
        "arm": "context",
        "model": MODEL,
        "reads_l19": reads,
        "thresholds": {"tier1_qualification": c.TF_QUALIFICATION_GAP, "tier2_kill": c.TF_KILL_GAP},
    }
    tf_sub, op = reads["tf_on_companion_subset"], reads["op_companion"]
    if tf_sub is None or op is None:
        reason = (
            "companion halted (rc=23 usable-floor miss / no usable companion convs) — "
            "TF headline proceeds, calibration N/A (plan v8 §4.5)"
            if companion_halted
            else "TF-on-companion / companion cell missing (smoke skip)"
        )
        # Production: ONLY a matched-recorded companion halt excuses missing
        # cells; a missing cell beside a LIVE companion is fits/refit drift.
        assert smoke or companion_halted, (
            "TF-on-companion / companion cell JSON missing in production with a "
            "non-halted companion — fits/matched-row drift"
        )
        payload["calibration"] = {
            "skipped": reason,
            "companion": "halted" if companion_halted else "missing_at_smoke_n",
        }
        print(f"[matched-row] TF/op calibration skipped: {reason}", flush=True)
    else:
        gap = tf_sub["r2_l19"] - op["r2_l19"]
        tier1 = gap > c.TF_QUALIFICATION_GAP
        tier2 = (op["r2_l19"] < 0.0) and (gap > c.TF_KILL_GAP)
        payload["calibration"] = {
            "tf_minus_op_gap_matched_subset": float(gap),
            "tier1_qualification": bool(tier1),
            "tier2_tf_distorted": bool(tier2),
            "note": (
                "tier1: companion read accompanies the TF headline as a qualification; "
                "tier2: TF cell reported with a TF-DISTORTED qualifier and the companion "
                "demoted to primary for this arm (plan v8 §7 — analyzer-side labels; "
                "the gap also bundles any residual pre-slot construction inflation)"
            ),
        }
        print(
            f"[matched-row] TF/op calibration: gap={gap:+.4f} tier1={tier1} tier2={tier2}",
            flush=True,
        )
    # Assumption-13 monotonicity report (subset R² <= full R² expected for r1)
    if reads["r1_matched"] and reads["r1_full"]:
        payload["r1_subset_vs_full"] = {
            "matched_l19": reads["r1_matched"]["r2_l19"],
            "full_l19": reads["r1_full"]["r2_l19"],
            "subset_leq_full": bool(reads["r1_matched"]["r2_l19"] <= reads["r1_full"]["r2_l19"]),
        }
    c.write_json(matched_out / "tf_op_calibration.json", payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--eval-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <eval-dir>/matched_row")
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=fc.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=fc.N_BOOTSTRAP)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert c.HAS_R4, (
        f"matched_row_refits requires EPM_I1345_VARIANT in {c.PAIRED_STORIES_VARIANTS} "
        f"(got {c.VARIANT!r})"
    )
    out_dir = args.out_dir or (args.eval_dir / "matched_row")
    preds_dir = args.preds_dir or (c.PREDS_CACHE_DIR / "matched_row")
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = load_matched(args.matched_dir)
    r4cfg = matched.get("per_model_r4_pair", {}).get(MODEL)
    if r4cfg is None:
        msg = f"no per_model_r4_pair entry for {MODEL} in matched_subsets.json"
        if args.smoke:
            print(f"[matched-row][smoke] SKIP phase: {msg} (r4 pair build smoke-skipped)")
            return
        raise RuntimeError(f"{msg} — run matchedn after extract_r4_tf (plan v8 §4)")

    cells, allow = matched_row_cells(r4cfg)
    print(
        f"[matched-row] {len(cells)} comparator cells on n={r4cfg['n']} r4-kept convs "
        f"(companion n={r4cfg.get('n_op', 0)})",
        flush=True,
    )
    run_cells(
        args.turnstore_dir,
        out_dir,
        preds_dir,
        cells,
        matched,
        n_folds=args.folds,
        seed=args.seed,
        null_draws=args.null_draws,
        n_boot=args.n_boot,
        smoke=args.smoke,
        allowlist_fn=lambda cell: allow[cell["cell_id"]],
    )
    # Companion demotion from the matched record (rc=23 lane / empty subset):
    # excuses the missing companion cells + writes companion: halted.
    tf_op_calibration(
        args.eval_dir,
        out_dir,
        smoke=args.smoke,
        companion_halted=not r4cfg.get("op_companion_convs"),
    )
    print("[done] matched-row comparator refits complete", flush=True)


if __name__ == "__main__":
    main()
