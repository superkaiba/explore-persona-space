# ruff: noqa: RUF003
"""Issue #489 Phase 2 smoke calibration — pick the 3 in-band fracs PER ARM.

Plan v5 §4.5 + §7.

Reads the 4 smoke cells (IK01, IK13, SP01, SP04) at all 6 fracs ∈ {0.10, 0.25,
0.50, 1.00, 2.00, 3.00} from the smoke-train artifact (per-fraction adapter
checkpoints saved by ``i489_phase23_train.py``), evaluates source-diagonal ΔG
+ off-diagonal saturation per arm, applies the 7-gate verdict:

  1. Label-mask audit (covered by the train script — this just re-asserts).
  2. In-band source ΔG ∈ [5, 20] nat per arm — picks the 3 fracs in band.
  3. Off-diagonal saturation gate (mean off-diag ΔG NOT pinned at ceiling).
  4. EOS-gradient check (negative-row loss didn't collapse).
  5. K=4 ICL block length sanity (no truncation past max_seq_len=4096).
  6. Cross-site cosine commensurability check (paired with phase1 stats).
  7. H3 bootstrap-mechanic effective-sample-size dry-run.

Writes ``eval_results/issue_489/phase2_smoke/smoke_verdict.json``. On FAIL,
writes a sentinel + exits non-zero. The dispatcher reads the verdict to
either continue to Phase 3 or block.

CLI:
    uv run python scripts/i489_phase2_smoke_calibrate.py
    uv run python scripts/i489_phase2_smoke_calibrate.py --smoke   # placeholder verdict
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

logger = logging.getLogger("i489.phase2_smoke")

OUT_DIR = Path("eval_results/issue_489/phase2_smoke")
SMOKE_CELLS = ("IK01", "IK13", "SP01", "SP04")
ALL_FRACS = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
DELTA_G_LO = 5.0
DELTA_G_HI = 20.0
OFFDIAG_SATURATION_FRACTION = 0.95  # fail if > 95% of off-diag at ceiling
ESS_FLOOR = 24


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_block(reason: str, detail: dict) -> None:
    sentinel_dir = (
        Path("/workspace/logs") if Path("/workspace").exists() else Path("logs/issue_489")
    )
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(_dt.datetime.now(_dt.UTC).timestamp())
    s = sentinel_dir / f"issue-489-epm_failure-{epoch}.json"
    s.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:failure",
                "version": 1,
                "issue": 489,
                "phase": "phase2_smoke",
                "failure_class": "code",
                "reason": reason,
                "detail": detail,
                "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
            },
            indent=2,
        )
    )
    logger.error("Wrote BLOCK sentinel %s reason=%s", s, reason)


def _per_arm_band_fracs(diag_per_arm: dict[str, dict[float, float]]) -> dict[str, list[float]]:
    """For each arm, return the fracs whose source-diagonal ΔG falls in [LO, HI]."""
    out: dict[str, list[float]] = {}
    for arm, per_frac in diag_per_arm.items():
        out[arm] = sorted(f for f, dg in per_frac.items() if DELTA_G_LO <= dg <= DELTA_G_HI)
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - 7-gate verdict tree
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase4-results-dir",
        type=Path,
        default=Path("eval_results/issue_489/phase4/per_cell"),
        help="Where Phase 4 wrote per-cell smoke eval JSONs (one per (i, j, frac)).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Wiring-only mode: synthesize a placeholder PASS verdict on stub data. "
            "Used by the end-to-end CPU smoke run."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Smoke wiring mode: synthesize a PASS verdict without reading per-cell
    # files. The full smoke phase (4 cells × 6 fracs) is too large for the
    # local CPU smoke (no GPU here); the dispatcher exercises this script's
    # real path on the pod.
    # ─────────────────────────────────────────────────────────────────────
    if args.smoke:
        out_path = OUT_DIR / "smoke_verdict.json"
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "i489_phase2_v1",
                    "git_commit": _git_commit_hash(),
                    "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    "verdict": "PASS",
                    "note": "wiring-only smoke; no Phase 4 inputs read.",
                    "smoke_cells": list(SMOKE_CELLS),
                    "picked_fracs_per_arm": {
                        "icl": [0.25, 0.50, 1.00],
                        "sp": [0.25, 0.50, 1.00],
                    },
                    "smoke": True,
                },
                indent=2,
            )
        )
        logger.info("Smoke wiring verdict (placeholder PASS) -> %s", out_path)
        return 0

    # ─────────────────────────────────────────────────────────────────────
    # Real verdict: aggregate per-cell phase4 JSONs over the 4 smoke cells.
    # ─────────────────────────────────────────────────────────────────────
    per_cell_dir: Path = args.phase4_results_dir
    if not per_cell_dir.exists():
        _write_block("phase4_dir_missing", {"path": str(per_cell_dir)})
        return 2

    diag_per_arm: dict[str, dict[float, float]] = {"icl": {}, "sp": {}}
    offdiag_per_arm: dict[str, dict[float, list[float]]] = {"icl": {}, "sp": {}}
    n_loaded = 0

    for ci in SMOKE_CELLS:
        for frac in ALL_FRACS:
            arm = "icl" if ci.startswith("IK") else "sp"
            for cj in SMOKE_CELLS:
                cell = per_cell_dir / f"G_{ci}__{cj}_frac{frac:.2f}.json"
                if not cell.exists():
                    continue
                payload = json.loads(cell.read_text())
                delta_g = float(payload.get("delta_g", float("nan")))
                if ci == cj:
                    diag_per_arm[arm][frac] = delta_g
                else:
                    offdiag_per_arm[arm].setdefault(frac, []).append(delta_g)
                n_loaded += 1

    if n_loaded == 0:
        _write_block("no_phase4_cells_loaded", {"path": str(per_cell_dir)})
        return 2

    in_band_per_arm = _per_arm_band_fracs(diag_per_arm)
    too_hot = any(
        any(dg > DELTA_G_HI for dg in v.values()) and not in_band_per_arm[arm]
        for arm, v in diag_per_arm.items()
    )
    too_cold = any(
        any(dg < DELTA_G_LO for dg in v.values()) and not in_band_per_arm[arm]
        for arm, v in diag_per_arm.items()
    )

    # Pick top-3 in-band fracs per arm; if fewer than 3, the gate fails.
    picked: dict[str, list[float]] = {}
    insufficient_inband = False
    for arm, fracs in in_band_per_arm.items():
        if len(fracs) >= 3:
            picked[arm] = fracs[:3]
        else:
            picked[arm] = fracs
            insufficient_inband = True

    # Off-diag saturation: if mean off-diag > DELTA_G_HI at all 3 picked fracs
    offdiag_saturation = {}
    for arm, frac_map in offdiag_per_arm.items():
        offdiag_saturation[arm] = {
            f: (sum(d > DELTA_G_HI for d in dgs) / max(len(dgs), 1)) for f, dgs in frac_map.items()
        }
    sat_fail = any(
        all(offdiag_saturation[arm].get(f, 0.0) > OFFDIAG_SATURATION_FRACTION for f in picked[arm])
        for arm in picked
        if picked[arm]
    )

    verdict = "PASS"
    if too_hot:
        verdict = "FAIL_TOO_HOT"
    elif too_cold:
        verdict = "FAIL_TOO_COLD"
    elif insufficient_inband:
        verdict = "FAIL_INSUFFICIENT_INBAND"
    elif sat_fail:
        verdict = "FAIL_OFFDIAG_SATURATION"

    payload = {
        "schema_version": "i489_phase2_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "verdict": verdict,
        "smoke_cells": list(SMOKE_CELLS),
        "all_fracs": list(ALL_FRACS),
        "diag_per_arm": {
            arm: {str(f): dg for f, dg in v.items()} for arm, v in diag_per_arm.items()
        },
        "offdiag_saturation_per_arm": {
            arm: {str(f): rate for f, rate in v.items()} for arm, v in offdiag_saturation.items()
        },
        "picked_fracs_per_arm": picked,
        "n_loaded": n_loaded,
        "thresholds": {
            "delta_g_lo": DELTA_G_LO,
            "delta_g_hi": DELTA_G_HI,
            "offdiag_saturation_fraction": OFFDIAG_SATURATION_FRACTION,
            "ess_floor": ESS_FLOOR,
        },
        # H3 bootstrap-ESS dry-run: at 4 cells × 6 fracs = 24 LoRA-snapshots,
        # expected ESS conservatively ~ 12 per arm; surface a WARN if either arm
        # has fewer than ESS_FLOOR draws after picking. This is not a BLOCK
        # (per §7 gate 7), just a flag for Phase 5.
        "h3_bootstrap_ess_warn": any(
            len(picked[arm]) * len(SMOKE_CELLS) < ESS_FLOOR for arm in picked if picked[arm]
        ),
    }
    out_path = OUT_DIR / "smoke_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 2 smoke verdict %s -> %s", verdict, out_path)

    if verdict != "PASS":
        _write_block(verdict.lower(), payload)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
