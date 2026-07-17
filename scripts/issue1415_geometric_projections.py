"""Issue #1415 geometric-projection driver (round-2 Critical fix). CPU, VM-safe.

Produces the plan §6.5 primary deliverable
``eval_results/issue_1415/geometric_projections.json`` — the H1 headline DV
(plan §4.10 DV (a)) plus the pre-registered H3 matched-vs-cross test — from
the phase-1a captures + the phase-1e steered-completions V_a captures.

Per steered cell (phases 1c grid/retry/layers/allpos + 1d full), per READ
layer l of the sweep::

    shift(cell)[l]      = V_a_mean(steered cell)[l] - V_a(c)[l]
    target(pair)[l]     = V_a(c')[l] - V_a(c)[l]
    projection_cosine   = cos( shift[l], target[l] )                 # PRIMARY
    frac_of_anchor      = <shift[l], target[l]> / ||target[l]||^2    # companion

**Units contract:** the PRIMARY statistic is the COSINE — dimensionless, so it
is commensurate with the null battery's cosine draws
(``issue1415_null_battery.py``; this script asserts the bands file declares
``units == "cosine"`` before any H1 band comparison) and the constructed
geometric ceiling is cos(target, target) = 1.0, per the plan §4.5
normalization-anchor note ("the geometric DV reported as a fraction of its
constructed anchor" — ``frac_of_anchor`` is that magnitude-bearing companion,
also 1.0 at the anchor, reported but NOT the verdict statistic).

H1 rows (per extraction arm x pair, at the coherence-selected operating
alpha): the MATCHED-layer statistic — the cell steered AT layer l read at
layer l — across the 7 sweep layers, then MAX over layers (the identical
selection rule the null battery applies per draw; selection-symmetric).
Coherence-failed pairs (operating alpha None) get an EXCLUDED row with the
recorded reason (plan §8), so every registered row exists in the file.

H3 (plan §3): per arm, one-sided two-sample Welch t-test (matched > cross;
independent samples) + Wilcoxon rank-sum robustness companion over the H1
row statistics. The driver asserts len(matched rows) == 15 and
len(cross rows) == 13 against the file (``--expect-counts``, default
``15,13``; the tiny smoke overrides) BEFORE running the test; excluded rows
are dropped from the TEST with realized Ns recorded as a deviation.

r_B cells (arm ``rb_<trait>``) and all-positions cells are reported as
descriptive companion sections (not part of H1/H3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch — the #847 thread-cap hook binds at import time

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1415_analysis_common as common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1415_geometric_projections")

# Kept in sync with issue1415_run_phase1.CAPTURED_PHASES (pinned by
# tests/test_issue1415_analysis.py::test_captured_phases_in_sync — duplicated
# here so this CPU driver does not import the GPU driver module).
CAPTURED_PHASES = (
    "phase1c_grid",
    "phase1c_retry",
    "phase1c_layers",
    "phase1c_allpos",
    "phase1d_full",
)
H1_PHASES = ("phase1c_grid", "phase1c_retry", "phase1c_layers")
EXPECT_COUNTS_DEFAULT = "15,13"  # plan §3: 15 matched-query vs 13 cross-query pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations",
        help="phase-1a capture dir (<pair_id>.pt)",
    )
    ap.add_argument(
        "--steered-activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations_steered",
        help="phase-1e per-cell steered V_a capture dir (<cell_id>.pt, nested)",
    )
    ap.add_argument(
        "--cells",
        type=Path,
        default=common.REPO_ROOT / "eval_results" / "issue_1415" / "phase1" / "cells",
        help="phase-1 per-cell metadata dir",
    )
    ap.add_argument(
        "--alpha-selection",
        type=Path,
        default=common.REPO_ROOT
        / "eval_results"
        / "issue_1415"
        / "phase1"
        / "alpha_selection_1c.json",
    )
    ap.add_argument(
        "--pair-bank",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "pair_bank.json",
    )
    ap.add_argument(
        "--null-bands",
        type=Path,
        default=None,
        help="null_bands.json for the H1 band comparison (units must be cosine); "
        "omitted -> comparison skipped with a recorded note",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=common.REPO_ROOT / "eval_results" / "issue_1415" / "geometric_projections.json",
    )
    ap.add_argument(
        "--expect-counts",
        default=EXPECT_COUNTS_DEFAULT,
        help="'<n_matched>,<n_cross>' H3 row-count assert (plan §3; default 15,13)",
    )
    ap.add_argument("--primary-layer", type=int, default=common.PRIMARY_LAYER)
    return ap.parse_args(argv)


# ── per-cell projections ──────────────────────────────────────────────


def _cell_arm(meta: dict) -> str:
    if meta["phase"] == "phase1d_full":
        return f"rb_{meta['trait']}"
    return meta["extraction_arm"]


def load_cell_metas(cells_dir: Path) -> list[dict]:
    assert cells_dir.exists(), f"cells metadata dir missing: {cells_dir}"
    metas = [json.loads(p.read_text()) for p in sorted(cells_dir.rglob("*.json"))]
    metas = [m for m in metas if m.get("phase") in CAPTURED_PHASES]
    assert metas, f"no steered-cell metadata under {cells_dir} (phases {CAPTURED_PHASES})"
    return metas


def cell_projections(
    meta: dict,
    steered_dir: Path,
    pair: common.PairTensors,
) -> dict:
    """One per-cell row: projection_cosine + frac_of_anchor per READ layer."""
    cell_id = meta["cell_id"]
    path = steered_dir / f"{cell_id}.pt"
    assert path.exists(), (
        f"phase-1e steered V_a capture missing for {cell_id}: {path} — "
        "run the phase-1 driver's 1e capture phase first"
    )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    row: dict = {
        "cell_id": cell_id,
        "pair_id": meta["pair_id"],
        "phase": meta["phase"],
        "arm": _cell_arm(meta),
        "steer_layer": meta["layer"],
        "alpha": meta["alpha"],
        "all_positions": meta["all_positions"],
        "passes_gate": meta.get("passes_gate"),
        "excluded_reason": None,
        "per_read_layer": {},
    }
    if blob.get("all_empty"):
        row["excluded_reason"] = "all_empty_completions"
        return row
    assert list(blob["layers"]) == pair.layers, (cell_id, blob["layers"], pair.layers)
    v_a = blob["v_a_mean"].float()  # (L, H)
    row["n_empty_completions"] = int(blob.get("n_empty_completions", 0))
    target = pair.target_raw  # (L, H)
    tnorm = target.norm(dim=-1)  # (L,)
    assert torch.all(tnorm > 0), (cell_id, "degenerate zero answer target")
    shift = v_a - pair.v_a_c  # (L, H)
    snorm = shift.norm(dim=-1)  # (L,)
    dot = (shift * target).sum(dim=-1)  # (L,)
    for li, layer in enumerate(pair.layers):
        if float(snorm[li]) == 0.0:
            row["per_read_layer"][str(layer)] = {
                "projection_cosine": None,
                "reason": "degenerate_zero_shift",
            }
            continue
        row["per_read_layer"][str(layer)] = {
            "projection_cosine": float(dot[li] / (snorm[li] * tnorm[li])),
            "frac_of_anchor": float(dot[li] / tnorm[li] ** 2),
            "shift_norm": float(snorm[li]),
            "target_norm": float(tnorm[li]),
        }
    return row


def matched_layer_cos(row: dict) -> float | None:
    """The MATCHED-layer statistic: the cell steered at layer l, read at l."""
    rec = row["per_read_layer"].get(str(row["steer_layer"]))
    return None if rec is None else rec.get("projection_cosine")


# ── H1 rows (per extraction arm x pair, at the operating alpha) ───────


def build_h1(
    per_cell: list[dict],
    selection: dict,
    pair_types: dict[str, str],
    layers: list[int],
    null_bands: dict | None,
) -> dict:
    """Per (extraction arm, pair): matched-layer cosines across the sweep
    layers at the operating alpha; MAX over layers (selection-symmetric with
    the null battery); optional band comparison in shared cosine units."""
    by_key: dict[tuple, dict] = {}
    for row in per_cell:
        if row["phase"] not in H1_PHASES or row["all_positions"]:
            continue
        by_key[(row["arm"], row["pair_id"], row["alpha"], row["steer_layer"])] = row

    h1: dict[str, dict] = {}
    for arm in common.ARMS:
        rows: dict[str, dict] = {}
        for key, sel in selection.items():
            sel_arm, pid = key.split("/", 1)
            if sel_arm != arm:
                continue
            op = sel["operating_alpha"]
            out_row: dict = {
                "pair_type": pair_types[pid],
                "operating_alpha": op,
                "per_layer_matched_cos": {},
                "max_over_layers": None,
                "excluded_reason": None,
            }
            if op is None:
                out_row["excluded_reason"] = "coherence_failed_all_alpha"
                rows[pid] = out_row
                continue
            vals = []
            for layer in layers:
                cell = by_key.get((arm, pid, op, layer))
                cos = matched_layer_cos(cell) if cell is not None else None
                out_row["per_layer_matched_cos"][str(layer)] = cos
                if cos is not None:
                    vals.append(cos)
            out_row["n_layers_used"] = len(vals)
            if not vals:
                out_row["excluded_reason"] = "no_valid_layer_projection"
            else:
                out_row["max_over_layers"] = max(vals)
            if null_bands is not None and out_row["max_over_layers"] is not None:
                cmp = {}
                for battery, bands in null_bands["bands"].items():
                    band = bands["per_pair"][arm][pid]
                    cmp[battery] = {
                        "null_p97.5": band["p97.5"],
                        "exceeds_p97.5": out_row["max_over_layers"] > band["p97.5"],
                    }
                out_row["band_comparison"] = cmp
            rows[pid] = out_row
        h1[arm] = rows
    return h1


# ── H3 (plan §3): matched vs cross, Welch one-sided + rank-sum ────────


def run_h3(h1: dict, expect_matched: int, expect_cross: int) -> dict:
    from scipy import stats

    out: dict = {
        "test": (
            "one-sided two-sample Welch t-test (matched > cross; independent samples) "
            "+ Wilcoxon rank-sum robustness companion (plan §3)"
        ),
        "row_statistic": (
            "H1 max-over-layers matched-layer projection_cosine at the operating alpha"
        ),
        "expected_rows": {"matched": expect_matched, "cross": expect_cross},
        "per_arm": {},
    }
    for arm, rows in h1.items():
        matched = {pid: r for pid, r in rows.items() if r["pair_type"] == "matched"}
        cross = {pid: r for pid, r in rows.items() if r["pair_type"] == "cross"}
        # Plan §3 row-coverage assert: every registered row EXISTS in the file
        # (excluded rows carry their recorded reason but still count as rows).
        assert len(matched) == expect_matched and len(cross) == expect_cross, (
            f"H3 row-count assert failed for arm {arm!r}: "
            f"matched={len(matched)} (want {expect_matched}), "
            f"cross={len(cross)} (want {expect_cross}) — "
            "pass --expect-counts only for a deliberately smaller smoke bank"
        )
        m_vals = [r["max_over_layers"] for r in matched.values() if r["excluded_reason"] is None]
        c_vals = [r["max_over_layers"] for r in cross.values() if r["excluded_reason"] is None]
        excluded = sorted(
            pid for pid, r in {**matched, **cross}.items() if r["excluded_reason"] is not None
        )
        arm_out: dict = {
            "n_matched_rows": len(matched),
            "n_cross_rows": len(cross),
            "n_used_matched": len(m_vals),
            "n_used_cross": len(c_vals),
            "excluded_pairs": excluded,
            "matched_mean": (sum(m_vals) / len(m_vals)) if m_vals else None,
            "cross_mean": (sum(c_vals) / len(c_vals)) if c_vals else None,
        }
        if excluded:
            arm_out["deviation"] = (
                "coherence-excluded rows dropped from the test (plan §8); "
                "realized Ns recorded above"
            )
        if len(m_vals) >= 2 and len(c_vals) >= 2:
            t = stats.ttest_ind(m_vals, c_vals, equal_var=False, alternative="greater")
            w = stats.ranksums(m_vals, c_vals, alternative="greater")
            arm_out["welch_t"] = float(t.statistic)
            arm_out["welch_p_one_sided"] = float(t.pvalue)
            arm_out["ranksum_stat"] = float(w.statistic)
            arm_out["ranksum_p_one_sided"] = float(w.pvalue)
        else:
            arm_out["welch_p_one_sided"] = None
            arm_out["ranksum_p_one_sided"] = None
            arm_out["reason"] = "insufficient non-excluded rows for the two-sample tests"
        out["per_arm"][arm] = arm_out
    return out


# ── main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    expect_matched, expect_cross = (int(x) for x in args.expect_counts.split(","))

    bank = json.loads(args.pair_bank.read_text())
    pair_types = {p["pair_id"]: p["pair_type"] for p in bank["pairs"]}

    pairs = {p.pair_id: p for p in common.load_all_pairs(args.activations)}
    layers = next(iter(pairs.values())).layers
    assert args.primary_layer in layers, (args.primary_layer, layers)

    null_bands = None
    if args.null_bands is not None:
        null_bands = json.loads(args.null_bands.read_text())
        # Units guard (round-2 scale-comparability fix): band and observed
        # must share the cosine statistic — never compare across units.
        units = null_bands.get("units")
        if units != "cosine":
            raise RuntimeError(
                f"null bands at {args.null_bands} declare units={units!r}, need 'cosine' — "
                "re-run scripts/issue1415_null_battery.py (round-2 cosine statistic)"
            )

    metas = load_cell_metas(args.cells)
    per_cell = [cell_projections(m, args.steered_activations, pairs[m["pair_id"]]) for m in metas]
    logger.info(
        "projected %d steered cells (%d excluded)",
        len(per_cell),
        sum(r["excluded_reason"] is not None for r in per_cell),
    )

    selection = json.loads(args.alpha_selection.read_text())["selection"]
    h1 = build_h1(per_cell, selection, pair_types, layers, null_bands)
    h3 = run_h3(h1, expect_matched, expect_cross)

    out = {
        "statistic": (
            "projection_cosine = cos(V_a(steered) - V_a(c), V_a(c') - V_a(c)) per read layer; "
            "H1 rows = matched-layer (steer layer == read layer) cosine at the operating alpha, "
            "MAX over sweep layers (selection-symmetric with the null battery)"
        ),
        "units": "cosine",
        "anchor_note": (
            "constructed geometric ceiling = 1.0 (cos(target, target); plan §4.5 "
            "normalization-anchor note); frac_of_anchor is the magnitude-bearing companion"
        ),
        "layers": layers,
        "primary_layer": args.primary_layer,
        "null_bands_file": str(args.null_bands) if args.null_bands else None,
        "null_band_comparison": (
            "per-row band_comparison vs per-pair p97.5 (shared cosine units)"
            if null_bands is not None
            else "SKIPPED — no --null-bands provided (run issue1415_null_battery.py first)"
        ),
        "h1": h1,
        "h3": h3,
        "per_cell": per_cell,
        "repro": common.repro_meta("issue1415_geometric_projections"),
    }
    common.write_json_atomic(args.out_json, out)
    logger.info(
        "wrote %s (%d per-cell rows; h3 welch p per arm: %s)",
        args.out_json,
        len(per_cell),
        {a: v["welch_p_one_sided"] for a, v in h3["per_arm"].items()},
    )


if __name__ == "__main__":
    main()
